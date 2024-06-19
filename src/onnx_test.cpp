
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <chrono>

using std::cout, std::endl;

int main() {
    // Allocate ONNXRuntime session
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Env env = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};
    Ort::Session session{env, ORT_TSTR("models/whistlenet.onnx"), Ort::SessionOptions{nullptr}};

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    std::cout << "Number of input nodes: " << numInputNodes << "\n";
    std::cout << "Number of output nodes: " << numOutputNodes << "\n";
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    std::array<float, 513> input{};
    std::array<int64_t, 3> input_shape{1,1,513};
    std::array<float, 2> output{0,0};
    std::array<int64_t, 2> output_shape{1,2};


    for(int i = 0; i < 2; i++) {
        // Allocate model inputs: fill in shape and size
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());

        // Allocate model outputs: fill in shape and size
        Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), output_shape.data(), output_shape.size());

        // Run the model
        auto start = std::chrono::high_resolution_clock::now();
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
        for(int j = 0; j < 2; j++) {
            std::cout << output[j] << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count();
        std::cout << "Run time: " << seconds << "microseconds\n";
    }
}