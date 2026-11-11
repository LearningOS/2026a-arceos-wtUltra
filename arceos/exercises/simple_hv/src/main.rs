#![cfg_attr(feature = "axstd", no_std)]
#![cfg_attr(feature = "axstd", no_main)]
#![feature(asm_const)]
#![feature(riscv_ext_intrinsics)]

#[cfg(feature = "axstd")]
extern crate axstd as std;
extern crate alloc;
#[macro_use]
extern crate axlog;

mod task;
mod vcpu;
mod regs;
mod csrs;
mod sbi;
mod loader;

use vcpu::VmCpuRegisters;
use riscv::register::{scause, sstatus, stval};
use csrs::defs::hstatus;
use tock_registers::LocalRegisterCopy;
use csrs::{RiscvCsrTrait, CSR};
use vcpu::_run_guest;
use sbi::SbiMessage;
use loader::load_vm_image;
use axhal::mem::{PhysAddr, PAGE_SIZE_4K};
use axhal::paging::MappingFlags;
use crate::regs::GprIndex;
use crate::regs::GprIndex::{A0, A1};

const VM_ENTRY: usize = 0x8020_0000;

#[cfg_attr(feature = "axstd", no_mangle)]
fn main() {
    ax_println!("Hypervisor ...");

    // A new address space for vm.
    let mut uspace = axmm::new_user_aspace().unwrap();

    // Load vm binary file into address space.
    if let Err(e) = load_vm_image("/sbin/skernel2", &mut uspace) {
        panic!("Cannot load app! {:?}", e);
    }

    // Map a page at guest address 0 and store the shutdown word the guest loads with ld a0, 64(zero).
    setup_guest_param_page(&mut uspace);
    // Setup context to prepare to enter guest mode.
    let mut ctx = VmCpuRegisters::default();
    prepare_guest_context(&mut ctx);

    // Setup pagetable for 2nd address mapping.
    let ept_root = uspace.page_table_root();
    prepare_vm_pgtable(ept_root);

    // Kick off vm and wait for it to exit.
    while !run_guest(&mut ctx) {
    }

    panic!("Hypervisor ok!");
}

fn prepare_vm_pgtable(ept_root: PhysAddr) {
    let hgatp = 8usize << 60 | usize::from(ept_root) >> 12;
    unsafe {
        core::arch::asm!(
            "csrw hgatp, {hgatp}",
            hgatp = in(reg) hgatp,
        );
        core::arch::riscv64::hfence_gvma_all();
    }
}

/// Map a page at guest address 0 and store the guest shutdown parameter
/// word (0x6688) at offset 64 of that page, i.e. at guest address 0x40.
///
/// The guest loads the word with ld a0, 64(zero), so its address is 0x40.
/// The loader only maps the page at VM_ENTRY, so this page must be mapped
/// before the first entry, and before prepare_vm_pgtable installs the
/// second stage page table and flushes the G stage TLB. The guest therefore
/// really reads the word from guest memory; the host only supplies the word
/// as it would for any guest boot parameter.
fn setup_guest_param_page(uspace: &mut axmm::AddrSpace) {
    const GUEST_PARAM_ADDR: usize = 0;
    const SHUTDOWN_PARAM: u64 = 0x6688;
    uspace
        .map_alloc(
            GUEST_PARAM_ADDR.into(),
            PAGE_SIZE_4K,
            MappingFlags::READ | MappingFlags::WRITE | MappingFlags::USER,
            true,
        )
        .expect("Failed to map guest param page");
    uspace
        .write((GUEST_PARAM_ADDR + 64).into(), &SHUTDOWN_PARAM.to_le_bytes())
        .expect("Failed to write guest param");
}

/// Emulate a guest read of the mhartid CSR.
///
/// The guest reads its hart id with csrr rd, mhartid, which is a CSRRS
/// instruction (funct3 = 0b010, rs1 = x0, a pure read) with csr = 0xF14.
/// In VS mode the access traps as an illegal instruction, with the
/// faulting instruction word in stval. If inst is such an access, write
/// the virtual hart id (0x1234) into the destination register, advance
/// sepc past the (4 byte) instruction, and return true. Otherwise return
/// false so the caller can report the illegal instruction.
fn emulate_mhartid_csr(ctx: &mut VmCpuRegisters, inst: u32) -> bool {
    const OPCODE_SYSTEM: u32 = 0x73;
    const FUNCT3_CSRRS: u32 = 0b010;
    const CSR_MHARTID: u32 = 0xf14;
    const VIRT_HART_ID: usize = 0x1234;
    if inst & 0x7f != OPCODE_SYSTEM {
        return false;
    }
    let funct3 = (inst >> 12) & 0x7;
    let csr = (inst >> 20) & 0xfff;
    if funct3 != FUNCT3_CSRRS || csr != CSR_MHARTID {
        return false;
    }
    let rd = (inst >> 7) & 0x1f;
    if rd != 0 {
        let reg = match GprIndex::from_raw(rd) {
            Some(reg) => reg,
            None => return false,
        };
        ctx.guest_regs.gprs.set_reg(reg, VIRT_HART_ID);
    }
    // A CSR instruction is a 4 byte instruction (bits [1:0] == 0b11);
    // a 2 byte compressed instruction would advance by 2 instead.
    ctx.guest_regs.sepc += if inst & 0x3 == 0x3 { 4 } else { 2 };
    true
}

fn run_guest(ctx: &mut VmCpuRegisters) -> bool {
    unsafe {
        _run_guest(ctx);
    }

    vmexit_handler(ctx)
}

#[allow(unreachable_code)]
fn vmexit_handler(ctx: &mut VmCpuRegisters) -> bool {
    use scause::{Exception, Trap};

    let scause = scause::read();
    match scause.cause() {
        Trap::Exception(Exception::VirtualSupervisorEnvCall) => {
            let sbi_msg = SbiMessage::from_regs(ctx.guest_regs.gprs.a_regs()).ok();
            ax_println!("VmExit Reason: VSuperEcall: {:?}", sbi_msg);
            if let Some(msg) = sbi_msg {
                match msg {
                    SbiMessage::Reset(_) => {
                        let a0 = ctx.guest_regs.gprs.reg(A0);
                        let a1 = ctx.guest_regs.gprs.reg(A1);
                        ax_println!("a0 = {:#x}, a1 = {:#x}", a0, a1);
                        assert_eq!(a0, 0x6688);
                        assert_eq!(a1, 0x1234);
                        ax_println!("Shutdown vm normally!");
                        return true;
                    },
                    _ => todo!(),
                }
            } else {
                panic!("bad sbi message! ");
            }
        },
        Trap::Exception(Exception::IllegalInstruction) => {
            // The first guest instruction is csrr a1, mhartid. A VS mode guest cannot read the M mode mhartid CSR, so the access traps as an illegal instruction with the faulting instruction word in stval. Emulate the read: the hypervisor assigns this guest the virtual hart id 0x1234, then resume after the instruction. Any other illegal instruction is still fatal.
            let inst = stval::read() as u32;
            if emulate_mhartid_csr(ctx, inst) {
                return false;
            }
            panic!("Bad instruction: {:#x} sepc: {:#x}",
                inst,
                ctx.guest_regs.sepc
            );
        },
        Trap::Exception(Exception::LoadGuestPageFault) => {
            panic!("LoadGuestPageFault: stval{:#x} sepc: {:#x}",
                stval::read(),
                ctx.guest_regs.sepc
            );
        },
        _ => {
            panic!(
                "Unhandled trap: {:?}, sepc: {:#x}, stval: {:#x}",
                scause.cause(),
                ctx.guest_regs.sepc,
                stval::read()
            );
        }
    }
    false
}

fn prepare_guest_context(ctx: &mut VmCpuRegisters) {
    // Set hstatus
    let mut hstatus = LocalRegisterCopy::<usize, hstatus::Register>::new(
        riscv::register::hstatus::read().bits(),
    );
    // Set Guest bit in order to return to guest mode.
    hstatus.modify(hstatus::spv::Guest);
    // Set SPVP bit in order to accessing VS-mode memory from HS-mode.
    hstatus.modify(hstatus::spvp::Supervisor);
    CSR.hstatus.write_value(hstatus.get());
    ctx.guest_regs.hstatus = hstatus.get();

    // Set sstatus in guest mode.
    let mut sstatus = sstatus::read();
    sstatus.set_spp(sstatus::SPP::Supervisor);
    ctx.guest_regs.sstatus = sstatus.bits();
    // Return to entry to start vm.
    ctx.guest_regs.sepc = VM_ENTRY;
}
