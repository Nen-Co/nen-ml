# NenML: Machine Learning Library

**Built from scratch following Nen principles: Static memory, zero-allocation, inline functions, static typing**

[![CI](https://github.com/Nen-Co/nen-ml/workflows/CI/badge.svg)](https://github.com/Nen-Co/nen-ml/actions)
[![Format Check](https://github.com/Nen-Co/nen-ml/workflows/Format%20Check/badge.svg)](https://github.com/Nen-Co/nen-ml/actions)

## 🎯 **Overview**

NenML is a high-performance machine learning library built from scratch in Zig, following the Nen way:
- ✅ **Static Memory**: Zero-allocation design with fixed memory pools
- ✅ **Inline Functions**: Performance-critical operations optimized
- ✅ **Static Typing**: Compile-time guarantees and error detection
- ✅ **Ecosystem Native**: Seamless integration with Nen components

## 🚀 **Features**

### **Core Components**
- **NenTensor**: Static memory tensor operations
- **NenNN**: Neural network layers built from scratch
- **NenMLFlow**: Workflow orchestration with ML nodes

### **Tensor Operations**
- Element-wise operations (add, multiply)
- Matrix multiplication
- Activation functions (relu, sigmoid, tanh, gelu, silu)
- Static memory pools for efficient reuse

### **Neural Network Layers**
- Linear layers with bias
- Layer normalization
- Dropout
- Token embeddings
- Multi-head attention
- Multiple activation functions

### **Workflow Orchestration**
- Node-based workflow management
- State tracking and monitoring
- Performance statistics
- Pre-built model templates

## 📦 **Installation**

### **Prerequisites**
- Zig 0.14.1 or later
- Nen ecosystem components (nen-io, nen-json)

### **Build**
```bash
# Clone the repository
git clone https://github.com/Nen-Co/nen-ml.git
cd nen-ml

# Build the library
zig build

# Run tests
zig build test

# Run examples
zig build run-tensor-example
zig build run-nn-example
zig build run-workflow-example
```

## 🎮 **Quick Start**

### **Basic Tensor Operations**
```zig
const nen_ml = @import("nen_ml");

// Create tensors
const shape = nen_ml.Shape.init(&.{ 2, 3 });
const tensor = nen_ml.Tensor.random(shape, .f32, 42);

// Element-wise operations
const ones = nen_ml.Tensor.ones(shape, .f32);
const sum = try tensor.add(ones);

// Matrix multiplication
const matrix_a = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 2, 2 }), .f32, 43);
const matrix_b = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 2, 2 }), .f32, 44);
const result = try matrix_a.matmul(matrix_b);

// Activation functions
const relu_result = tensor.relu();
const sigmoid_result = tensor.sigmoid();
```

### **Neural Network Layers**
```zig
// Linear layer
const input = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 2, 3 }), .f32, 45);
const linear = nen_ml.Linear.init(3, 4, .f32);
const output = try linear.forward(input);

// Activation functions
const relu = nen_ml.Activation.relu;
const activated = relu.forward(output);

// Layer normalization
const norm = nen_ml.LayerNorm.init(nen_ml.Shape.init(&.{ 4 }), .f32);
const normalized = norm.forward(output);

// Dropout
const dropout = nen_ml.Dropout.init(0.1);
const dropped = dropout.forward(normalized);
```

### **Workflow Orchestration**
```zig
const allocator = std.heap.page_allocator;

// Create workflow
const flow = try nen_ml.NenMLFlow.init(allocator);
defer flow.deinit();

// Add nodes
const node = try allocator.create(nen_ml.MLNode);
node.* = nen_ml.MLNode.init("linear_layer", .linear_layer);
node.setShapes(input_shape, output_shape);
try flow.addNode(node);

// Execute workflow
try flow.execute();

// Get statistics
const stats = flow.getStats();
log.info("Success rate: {d:.2}%", .{ stats.getSuccessRate() * 100.0 });
```

### **Pre-built Models**
```zig
// Linear model
const linear_model = try nen_ml.createLinearModel(allocator, 784, 128, 10);
defer linear_model.deinit();
try linear_model.execute();

// Transformer block
const transformer = try nen_ml.createTransformerBlock(allocator, 512, 8);
defer transformer.deinit();
try transformer.execute();
```

## 🏗️ **Architecture**

### **Design Principles**
1. **Static Memory**: All tensors use fixed-size buffers (1KB per tensor)
2. **Zero Allocation**: No heap allocations in hot paths
3. **Inline Functions**: Performance-critical operations are inline
4. **Type Safety**: Compile-time error detection and guarantees
5. **Ecosystem Integration**: Native support for Nen components

### **Memory Management**
```zig
// Static tensor with fixed buffer
pub const Tensor = struct {
    data: [1024]u8, // 1KB fixed buffer
    data_len: usize,
    shape: Shape,
    dtype: DataType,
};

// Memory pool for efficient reuse
pub const TensorPool = struct {
    tensors: [16]Tensor, // Pool of 16 tensors
    used: [16]bool,
};
```

### **Performance Features**
- **Inline Functions**: Critical operations optimized at compile time
- **Static Memory**: Predictable memory usage and cache locality
- **Type Safety**: Compile-time error detection
- **Zero Allocation**: No heap overhead in hot paths

## 📊 **Performance**

### **Benchmarks**
- **Tensor Operations**: >1M ops/sec
- **Neural Network**: >100K ops/sec
- **Workflow**: >10K ops/sec

### **Memory Usage**
- **Per Tensor**: 1KB static buffer
- **Tensor Pool**: 16KB total (16 tensors)
- **No Heap Allocation**: Zero garbage collection overhead

## 🔧 **Development**

### **Project Structure**
```
nen-ml/
├── src/
│   ├── lib.zig          # Main library exports
│   ├── tensor.zig       # Tensor operations
│   ├── nn.zig          # Neural network layers
│   ├── workflow.zig    # Workflow orchestration
│   └── main.zig        # Main executable
├── examples/
│   ├── tensor_demo.zig  # Tensor operations demo
│   ├── nn_demo.zig     # Neural network demo
│   └── workflow_demo.zig # Workflow demo
├── tests/
│   └── performance.zig  # Performance tests
└── build.zig           # Build configuration
```

### **Testing**
```bash
# Run all tests
zig build test-all

# Run unit tests
zig build test

# Run performance tests
zig build test-performance

# Run examples
zig build run-examples
```

### **CI/CD**
- Automated testing on Linux, macOS, Windows
- Format checking with `zig fmt`
- Performance benchmarking
- Cross-platform compatibility

## 🌐 **Ecosystem Integration**

### **Nen Components**
- **NenDB**: Store model metadata and training history
- **NenCache**: Cache inference results and model weights
- **NenIO**: Handle model file I/O operations
- **NenJSON**: Serialize model configurations and results
- **NenNet**: Enable distributed training and inference

### **Example Integration**
```zig
// Store model metadata in NenDB
try db.storeModelMetadata("model", metadata);

// Cache inference results in NenCache
try cache.storeInferenceResult("input", result);

// Use NenIO for model files
const model_data = try io.readFile("model.pt");

// Serialize with NenJSON
const config = try json.parse(config_data);
```

## 🚀 **Roadmap**

### **Phase 1: Core Completion**
- [x] Basic tensor operations
- [x] Neural network layers
- [x] Workflow orchestration
- [ ] Convolutional layers
- [ ] Pooling operations
- [ ] Batch normalization

### **Phase 2: Model Support**
- [ ] Model loading and saving
- [ ] Weight management
- [ ] Inference engine
- [ ] Example models (MNIST, simple transformers)

### **Phase 3: Production Features**
- [ ] Quantization (INT8, FP16)
- [ ] Performance optimization
- [ ] Monitoring and debugging
- [ ] Multi-node support

### **Phase 4: Ecosystem Integration**
- [ ] Full NenDB integration
- [ ] Complete NenCache integration
- [ ] NenNet distributed computing
- [ ] Cross-language bindings

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
git clone https://github.com/Nen-Co/nen-ml.git
cd nen-ml
zig build test
```

### **Code Style**
- Follow Zig style guidelines
- Use `zig fmt` for formatting
- Write comprehensive tests
- Follow Nen principles (static memory, zero-allocation)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Built following the **Nen way** principles
- Inspired by modern ML frameworks but designed for performance
- Part of the Nen ecosystem for production ML workloads

---

**NenML**: Machine learning the Nen way - static memory, zero-allocation, maximum performance! 🚀
