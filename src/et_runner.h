#pragma once

#include <cassert>
#include <vector>
#include <memory>
#include <iostream>
#include <chrono>

#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/error.h>
#include <executorch/util/util.h>
#include <executorch/extension/evalue_util/print_evalue.h>


using torch::executor::runtime_init;
static constexpr torch::executor::LogLevel Info = torch::executor::LogLevel::Info;
using torch::executor::util::FileDataLoader;
using torch::executor::Result;
using torch::executor::Program;
using torch::executor::MethodMeta;
using torch::executor::MemoryManager;
using torch::executor::MemoryAllocator;
using torch::executor::Span;
using torch::executor::HierarchicalAllocator;
using torch::executor::Method;
using torch::executor::util::prepare_input_tensors;
using torch::executor::Error;
using torch::executor::EValue;


class Runner {
    private:
        Result<Program> loadProgram(const char* model_path);

        const char* method_name = nullptr;
        Result<MethodMeta> getMethodMeta(Program& program);

        double run(Method& method);

    public:
        Runner(const char* model_path);
        ~Runner() = default;
};