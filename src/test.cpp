#include "et_runner.h"
#include <iostream>

int main() {
    std::string path = "models/cnn.pte";
    const char* model_path = path.c_str();
    Runner runner = Runner(model_path);
    return 0;
} 
