


#include "shuriken/parser/Dex/dex_types.h"
#include "transform/lifter.h"
#include <mlir/IR/OpDefinition.h>
#include <string_view>

using namespace shuriken::MjolnIR;
void Lifter::gen_instruction(shuriken::disassembler::dex::Instruction21c *instr) {
    auto op_code = static_cast<DexOpcodes::opcodes>(instr->get_instruction_opcode());

    auto location = mlir::FileLineColLoc::get(&context, module_name, instr->get_address(), 0);

    auto dest = instr->get_destination();

    switch (op_code) {
        case DexOpcodes::opcodes::OP_NEW_INSTANCE: {

            auto *cls = std::get<DVMType *>(instr->get_source_as_kind());

            auto cls_type = get_type(cls);

            auto gen_value = builder.create<::mlir::shuriken::MjolnIR::NewOp>(
                    location,
                    cls_type);

            writeVariable(current_basic_block, dest, gen_value);
        } break;
        case DexOpcodes::opcodes::OP_CONST_STRING: {

            std::string_view str_value = std::get<std::string_view>(instr->get_source_as_kind());
            // instr->get_source_str();
            auto str_ref = instr->get_source();

            auto gen_value = builder.create<::mlir::shuriken::MjolnIR::LoadString>(
                    location,
                    strObjectType,
                    str_value,
                    str_ref);

            writeVariable(current_basic_block, dest, gen_value);
        } break;
        case DexOpcodes::opcodes::OP_SGET:
        case DexOpcodes::opcodes::OP_SGET_WIDE:
        case DexOpcodes::opcodes::OP_SGET_OBJECT:
        case DexOpcodes::opcodes::OP_SGET_BOOLEAN:
        case DexOpcodes::opcodes::OP_SGET_BYTE:
        case DexOpcodes::opcodes::OP_SGET_CHAR:
        case DexOpcodes::opcodes::OP_SGET_SHORT: {
            auto *field = std::get<FieldID *>(instr->get_source_as_kind());

            std::string_view field_name = field->field_name();
            std::string_view field_class = field->field_class()->get_raw_type();

            auto dest_type = get_type(field->field_type());

            auto generated_value = builder.create<::mlir::shuriken::MjolnIR::LoadFieldOp>(
                    location,
                    field_name,
                    field_class,
                    dest_type);

            writeVariable(current_basic_block, dest, generated_value);
        } break;
        case DexOpcodes::opcodes::OP_SPUT:
        case DexOpcodes::opcodes::OP_SPUT_WIDE:
        case DexOpcodes::opcodes::OP_SPUT_OBJECT:
        case DexOpcodes::opcodes::OP_SPUT_BOOLEAN:
        case DexOpcodes::opcodes::OP_SPUT_BYTE:
        case DexOpcodes::opcodes::OP_SPUT_CHAR:
        case DexOpcodes::opcodes::OP_SPUT_SHORT: {
            auto *field = std::get<FieldID *>(instr->get_source_as_kind());

            std::string_view field_name = field->field_name();
            std::string_view field_class = field->field_class()->get_raw_type();

            auto reg = instr->get_destination();
            auto reg_value = readVariable(current_basic_block, current_method->get_basic_blocks(), reg);

            builder.create<::mlir::shuriken::MjolnIR::StoreFieldOp>(
                location,
                reg_value,
                field_name,
                field_class
            );   
        } break;
        default:
            throw exceptions::LifterException("Lifter::gen_instruction: Instruction21c not supported");
            break;
    }
}
