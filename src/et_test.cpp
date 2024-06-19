#include <executorch/extension/module/module.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <iostream>
#include <array>
#include <chrono>

using torch::executor::Module;
using torch::executor::TensorImpl;
using torch::executor::ScalarType; 
using torch::executor::Tensor;
using torch::executor::EValue;

int main() {
    Module model("models/whistlenet_quantized.pte", torch::executor::Module::MlockConfig::UseMlock);

    int iterations = 1000;
    double elapsed_time = 0;

    for (int i = 0; i < iterations; i++) {

        std::array<float, 513> input;
        for (int i = 0; i < 513; i++) {
            input[i] = 1.0;
            // input[i] = std::rand() / (float)RAND_MAX;
        }

        std::array<int32_t, 3> sizes{1, 1, 513};
        TensorImpl tensor(
            ScalarType::Float, 
            sizes.size(), 
            sizes.data(), 
            input.data()
        );

        auto start = std::chrono::high_resolution_clock::now();
        const auto result = model.execute("forward", {EValue(Tensor(&tensor))});
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        double seconds = duration.count();
        ET_LOG(
            Info,
            "inference took %f ms",
            duration
        );
        std::cout << result.get()[0] << std::endl;

        elapsed_time += seconds;

    }
    std::cout << "elapsed time: " << elapsed_time / static_cast<double>(iterations) << " ms" << std::endl;
} 
