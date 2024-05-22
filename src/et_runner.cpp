#include "et_runner.h"

Runner::Runner(const char* model_path) {
    assert(model_path != nullptr);

    runtime_init();
    Result<Program> program = loadProgram(model_path);
    Result<MethodMeta> method_meta = getMethodMeta(*program);

    // MemoryAllocator used to allocate runtime structures at Method load time. 
    // Things like Tensor metadata, the internal chain of instructions, and other runtime state come from this.
    uint8_t method_allocator_pool[4 * 1024U * 1024U]; // 4 MB
    MemoryAllocator method_allocator{MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};
    std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
    std::vector<Span<uint8_t>> planned_spans; // Passed to the allocator
    size_t num_memory_planned_buffers = method_meta->num_memory_planned_buffers();
    for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
        // .get() will always succeed because id < num_memory_planned_buffers.
        size_t buffer_size =
            static_cast<size_t>(method_meta->memory_planned_buffer_size(id).get());
        // ET_LOG(Info, "Setting up planned buffer %zu, size %zu.\n", id, buffer_size);
        planned_buffers.push_back(std::make_unique<uint8_t[]>(buffer_size));
        planned_spans.push_back({planned_buffers.back().get(), buffer_size});
    }

    // Planned Memory: A HierarchicalAllocator containing 1 or more memory spans 
    // where internal mutable tensor data buffers are placed.
    HierarchicalAllocator planned_memory({planned_spans.data(), planned_spans.size()});

    // Assemble all of the allocators into the MemoryManager that the Executor will use.
    MemoryManager memory_manager(&method_allocator, &planned_memory);
    ET_LOG(Info, "MemoryManager for %s is loaded.", method_name);

    Result<Method> method = program->load_method(method_name, &memory_manager);
    ET_CHECK_MSG(
        method.ok(),
        "Loading of method %s failed with status 0x%" PRIx32,
        method_name,
        (uint32_t)method.error());
    ET_LOG(Info, "Method for %s is loaded.", method_name);

    const double elapsed_time = run(*method);
    ET_LOG(
        Info,
        "inference took %f ms",
        elapsed_time
    );

}

double Runner::run(Method& method) {
    auto inputs = prepare_input_tensors(method);
    std::cout << inputs.ok() << std::endl;
    ET_CHECK_MSG(
        inputs.ok(),
        "Could not prepare inputs: 0x%" PRIx32,
        (uint32_t)inputs.error());
    ET_LOG(Info, "Inputs prepared.");

    // Run the model.
    auto before_exec = std::chrono::high_resolution_clock::now();
    Error status = method.execute();
    auto after_exec = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration_cast<std::chrono::milliseconds>(after_exec - before_exec).count();

    ET_CHECK_MSG(
        status == Error::Ok,
        "Execution of method %s failed with status 0x%" PRIx32,
        method_name,
        (uint32_t)status);
    printf("Model executed successfully.\n");

    // Print outputs
    std::vector<EValue> outputs(method.outputs_size());
    printf("%zu outputs: ", outputs.size());
    status = method.get_outputs(outputs.data(), outputs.size());
    ET_CHECK(status == Error::Ok);
    std::cout << torch::executor::util::evalue_edge_items(10);
    for (int i = 0; i < outputs.size(); ++i) {
        std::cout << "Output " << i << ": " << outputs[i] << std::endl;
    }

    return duration;
}

Result<Program> Runner::loadProgram(const char* model_path) {
    assert(model_path != nullptr);
    
    // Create a loader to get the data of the program file.
    Result<FileDataLoader> loader = FileDataLoader::from(model_path);
    ET_CHECK_MSG(
        loader.ok(),
        "FileDataLoader::from() failed: 0x%" PRIx32,
        (uint32_t)loader.error()
    );

    // Load the program
    Result<Program> program = Program::load(&loader.get());
    if (!program.ok()) {
        ET_LOG(Error, "Failed to parse model file %s", model_path);
    }
    ET_LOG(Info, "Model file %s is loaded.", model_path);

    return program;
}

Result<MethodMeta> Runner::getMethodMeta(Program& program) {

    // Method Name
    const auto method_name_result = program.get_method_name(0);
    ET_CHECK_MSG(method_name_result.ok(), "Program has no methods");
    method_name = *method_name_result;

    /* MethodMeta is a lightweight structure that lets us gather metadata
    information about a specific method. In this case we are looking to
    get the required size of the memory planned buffers for the method "forward".*/
    Result<MethodMeta> method_meta = program.method_meta(method_name);

    ET_CHECK_MSG(
        method_meta.ok(),
        "Failed to get method_meta for %s: 0x%" PRIx32,
        method_name,
        (uint32_t)method_meta.error()
    );

    ET_LOG(Info, "MethodMeta for %s is loaded.", method_name);
    return method_meta.get();
}