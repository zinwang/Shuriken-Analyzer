//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file dex_disassembler.cpp

#include "shuriken/disassembler/Dex/dex_disassembler.h"
#include "shuriken/common/logger.h"

using namespace shuriken::disassembler::dex;

DexDisassembler::DexDisassembler(parser::dex::Parser *parser)
    : parser(parser) {
    internal_disassembler = std::make_unique<Disassembler>(parser);
    linear_sweep.set_disassembler(internal_disassembler.get());
}

void DexDisassembler::disassemble_new_dex(parser::dex::Parser *new_parser) {
    this->parser = new_parser;
    internal_disassembler = std::make_unique<Disassembler>(parser);
    linear_sweep.set_disassembler(internal_disassembler.get());
    // finally call the disassembly
    disassembly_dex();
}

void DexDisassembler::set_disassembly_algorithm(disassembly_algorithm_t algorithm) {
    this->disassembly_algorithm = algorithm;
}

DisassembledMethod *DexDisassembler::get_disassembled_method(std::string method) {
    if (disassembled_methods.find(method) == disassembled_methods.end())
        return nullptr;
    return disassembled_methods[method].get();
}

DisassembledMethod *DexDisassembler::get_disassembled_method(std::string_view method) {
    if (disassembled_methods.find(method) == disassembled_methods.end())
        return nullptr;
    return disassembled_methods[method].get();
}

DexDisassembler::disassembled_methods_s_t &
DexDisassembler::get_disassembled_methods() {
    if (disassembled_methods_s.empty() || disassembled_methods_s.size() != disassembled_methods.size()) {
        for (const auto &entry: disassembled_methods)
            disassembled_methods_s.insert({entry.first, std::cref(*entry.second)});
    }
    return disassembled_methods_s;
}

DexDisassembler::disassembled_methods_t &
DexDisassembler::get_disassembled_methods_ownership() {
    return disassembled_methods;
}

void DexDisassembler::disassembly_dex() {

    log(LEVEL::INFO, "Starting disassembly of the DEX file");

    auto &classes = parser->get_classes();

    for (auto &class_def: classes.get_classdefs()) {
        auto &class_data_item = class_def.get_class_data_item();
        /// first disassemble the direct methods
        for (auto &method: class_data_item.get_direct_methods()) {
            disassemble_encoded_method(&method);
        }
        /// now the virtual methods
        for (auto &method: class_data_item.get_virtual_methods()) {
            disassemble_encoded_method(&method);
        }
    }

    log(LEVEL::INFO, "Finished method disassembly");
}

void DexDisassembler::disassemble_encoded_method(shuriken::parser::dex::EncodedMethod *method) {
    auto *code_item_struct = method->get_code_item();
    std::unique_ptr<DisassembledMethod> disassembled_method;
    std::vector<exception_data_t> exceptions_data;
    std::vector<std::unique_ptr<Instruction>> instructions;
    std::uint16_t n_of_registers = 0;

    if (code_item_struct != nullptr) {
        n_of_registers = code_item_struct->get_registers_size();
        exceptions_data = internal_disassembler->determine_exception(method);
        instructions = linear_sweep.disassembly(code_item_struct->get_bytecode());
    }

    disassembled_methods[method->getMethodID()->dalvik_name_format()] = std::make_unique<DisassembledMethod>(
            method->getMethodID(), n_of_registers, exceptions_data, instructions, method->get_flags());
}

std::vector<std::unique_ptr<Instruction>> DexDisassembler::disassembly_buffer(std::span<std::uint8_t> buffer) {
    return linear_sweep.disassembly(buffer);
}