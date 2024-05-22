
#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

using std::cout, std::endl;


int main() {
    // Allocate ONNXRuntime session
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Env env;
    Ort::Session session{env, ORT_TSTR("cnn.onnx"), Ort::SessionOptions{nullptr}};

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    std::cout << "Number of input nodes: " << numInputNodes << "\n";
    std::cout << "Number of output nodes: " << numOutputNodes << "\n";

    Ort::AllocatorWithDefaultOptions allocator;
    const char* inputName = session.GetInputNameAllocated(0, allocator).get();
    std::cout << "Input name: " << inputName << "\n";
    const char* outputName = session.GetOutputNameAllocated(0, allocator).get();
    std::cout << "Output name: " << outputName << "\n";

    // Allocate model inputs: fill in shape and size
    std::array<float, 784> input{};
    std::array<int64_t, 4> input_shape{1,1,28,28};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {inputName};

    // Allocate model outputs: fill in shape and size
    std::array<float, 10> output{};
    std::array<int64_t, 2> output_shape{10,1};
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), output_shape.data(), output_shape.size());
    const char* output_names[] = {outputName};

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    // Run the model
    // session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
}