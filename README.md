# Information

## Install

1. 
```
conda create -yn executorch python=3.11.0
conda activate executorch
./install_requirements.sh
```

- [Getting Started](https://pytorch.org/executorch/stable/getting-started-setup.html)
-[Building with CMake](https://pytorch.org/executorch/stable/runtime-build-and-cross-compilation.html)

## Toy Example

- [PyTorch Source](https://pytorch.org/executorch/stable/tutorials/export-to-executorch-tutorial.html)
- [Explanation of the three phases](https://pytorch.org/executorch/stable/ir-exir.html)

## Python Steps

1) Core ATen Dialect
    - [export()](https://pytorch.org/docs/2.1/export.html#torch.export.export) takes an arbitrary Python callable (an nn.Module, a function or a method) and produces a traced graph representing only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion, which can subsequently be executed with different outputs or serialized.
    - The traced graph does three things
        1) produces a normalized operator set consisting only of functional [Core ATen Operator Set](https://pytorch.org/docs/2.1/torch.compiler_ir.html) and user specified custom operators
        2) has eliminated all Python control flow and data structures (except for certain conditions)
        3) has the set of shape constraints needed to show that this normalization and control flow elimination is sound for a future input.
    - We will print the graph, represented through [Torch Fx](https://pytorch.org/docs/stable/fx.html#module-torch.fx), in a table to explicitly represent inputs, operations and outputs (more info [here](https://pytorch.org/docs/stable/fx.html#torch.fx.Node))
    - Internally, what happens is that we first convert to Pre-Autograd ATen Dialect
        - Trace a module before any pre-autograd decomposition is run.
        - Tracing means converting a PyTorch model into a more efficient, serialized format suitable for deployment in production or environments where Python might not be available.

            `
            from torch._export import capture_pre_autograd_graph
            pre_autograd_aten_dialect = capture_pre_autograd_graph(m, m.example_args)
            `

    - Somewhere (it is not clear if before or after the aten dialect) we quantize
        - Lower the precision, maintaining the accuracy high enough
        - Goal is to enhance memory and computational efficiency

            `
            from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
            from torch.ao.quantization.quantizer.xnnpack_quantizer import (
                get_symmetric_quantization_config,
                XNNPACKQuantizer,
            )

            quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
            prepared_graph = prepare_pt2e(pre_autograd_aten_dialect, quantizer)
            converted_graph = convert_pt2e(prepared_graph)
            `

2) Edge Dialect
    - to_edge() returns an EdgeProgramManager object, which contains the exported programs which will be placed on this device
        - DType specialization (to reduce binary size)
        - Scalar to tensor conversion
        - Converting all ops to the executorch.exir.dialects.edge namespace
    - Delegate to a Backend (optional)

3) Executorch Program

## Cpp information

### Memory allocation

One of the principles of ExecuTorch is giving users control over where the
memory used by the runtime comes from.

We need to define two different allocators.  

**First allocator: MemoryAllocator**: used to allocate runtime structures at Method load time.
Things like Tensor metadata, the internal chain of instructions, and other
runtime state come from this. This allocator is only used during loading a
method of the program, which will return an error if there was not enough
memory. The amount of memory required depends on the loaded method and the runtime code
itself. The amount of memory here is usually determined by running the method
and seeing how much memory is actually used, though it's possible to subclass
MemoryAllocator so that it calls malloc() under the hood (see
MallocMemoryAllocator). In this example we use a statically allocated memory pool.  

**Second Allocator: Planned Memory**: A HierarchicalAllocator containing 1 or more
memory spans where internal mutable tensor data buffers are placed. At Method
load time internal tensors have their data pointers assigned to various offsets
within. The positions of those offsets and the sizes of the arenas are
determined by memory planning ahead of time.