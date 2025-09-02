// NenML: Main executable demonstrating ML capabilities
// Built from scratch following Nen principles

const std = @import("std");
const nen_ml = @import("nen_ml");

const log = std.log.scoped(.nen_ml);

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    
    log.info("🚀 NenML - Machine Learning Library", .{});
    log.info("Built from scratch following Nen principles", .{});
    log.info("Static memory, zero-allocation, inline functions", .{});
    
    // Demo 1: Tensor Operations
    try demoTensorOperations();
    
    // Demo 2: Neural Network Layers
    try demoNeuralNetworkLayers();
    
    // Demo 3: Workflow Orchestration
    try demoWorkflowOrchestration();
    
    // Demo 4: Performance Benchmark
    try demoPerformanceBenchmark(allocator);
    
    log.info("✅ All demos completed successfully!", .{});
}

fn demoTensorOperations() !void {
    log.info("\n📊 Demo 1: Tensor Operations", .{});
    
    // Create tensors
    const shape = nen_ml.Shape.init(&.{ 2, 3 });
    const tensor = nen_ml.Tensor.random(shape, .f32, 42);
    
    log.info("Created tensor: shape={d}x{d}, dtype={s}", .{
        tensor.shape.getDim(0),
        tensor.shape.getDim(1),
        @tagName(tensor.dtype),
    });
    
    // Element-wise operations
    const ones = nen_ml.Tensor.ones(shape, .f32);
    
    const sum = try tensor.add(ones);
    const product = try tensor.multiply(ones);
    
    log.info("Tensor operations: add={d} elements, multiply={d} elements", .{
        sum.totalElements(),
        product.totalElements(),
    });
    
    // Matrix operations
    const matrix_shape = nen_ml.Shape.init(&.{ 2, 2 });
    const matrix_a = nen_ml.Tensor.random(matrix_shape, .f32, 43);
    const matrix_b = nen_ml.Tensor.random(matrix_shape, .f32, 44);
    
    const matmul_result = try matrix_a.matmul(matrix_b);
    
    log.info("Matrix multiplication: {d}x{d} @ {d}x{d} = {d}x{d}", .{
        matrix_a.shape.getDim(0), matrix_a.shape.getDim(1),
        matrix_b.shape.getDim(0), matrix_b.shape.getDim(1),
        matmul_result.shape.getDim(0), matmul_result.shape.getDim(1),
    });
    
    // Activation functions
    const relu_result = tensor.relu();
    const sigmoid_result = tensor.sigmoid();
    
    log.info("Activation functions: relu={d} elements, sigmoid={d} elements", .{
        relu_result.totalElements(),
        sigmoid_result.totalElements(),
    });
}

fn demoNeuralNetworkLayers() !void {
    log.info("\n🧠 Demo 2: Neural Network Layers", .{});
    
    // Linear layer
    const input_shape = nen_ml.Shape.init(&.{ 2, 3 });
    const input = nen_ml.Tensor.random(input_shape, .f32, 45);
    
    const linear = nen_ml.Linear.init(3, 4, .f32);
    const linear_output = try linear.forward(input);
    
    log.info("Linear layer: {d}x{d} -> {d}x{d}", .{
        input.shape.getDim(0), input.shape.getDim(1),
        linear_output.shape.getDim(0), linear_output.shape.getDim(1),
    });
    
    // Activation functions
    const relu = nen_ml.Activation.relu;
    const gelu = nen_ml.Activation.gelu;
    const silu = nen_ml.Activation.silu;
    
    _ = relu.forward(linear_output);
    _ = gelu.forward(linear_output);
    _ = silu.forward(linear_output);
    
    log.info("Activation functions: relu, gelu, silu applied", .{});
    
    // Layer normalization
    const norm_shape = nen_ml.Shape.init(&.{ 4 });
    const layer_norm = nen_ml.LayerNorm.init(norm_shape, .f32);
    const norm_output = layer_norm.forward(linear_output);
    
    log.info("Layer normalization: {d}x{d} -> {d}x{d}", .{
        linear_output.shape.getDim(0), linear_output.shape.getDim(1),
        norm_output.shape.getDim(0), norm_output.shape.getDim(1),
    });
    
    // Dropout
    const dropout = nen_ml.Dropout.init(0.1);
    _ = dropout.forward(norm_output);
    
    log.info("Dropout: rate=0.1 applied", .{});
    
    // Token embedding
    const vocab_size: usize = 1000;
    const embedding_dim: usize = 64;
    const embedding = nen_ml.TokenEmbedding.init(vocab_size, embedding_dim, .f32);
    
    const token_shape = nen_ml.Shape.init(&.{ 1, 5 }); // batch_size=1, seq_len=5
    const tokens = nen_ml.Tensor.random(token_shape, .i32, 46);
    const embedded = try embedding.forward(tokens);
    
    log.info("Token embedding: vocab={d}, dim={d}, output={d}x{d}x{d}", .{
        vocab_size, embedding_dim,
        embedded.shape.getDim(0), embedded.shape.getDim(1), embedded.shape.getDim(2),
    });
    
    // Multi-head attention
    const d_model: usize = 64;
    const num_heads: usize = 8;
    const attention = nen_ml.MultiHeadAttention.init(d_model, num_heads, .f32);
    
    const attention_input = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 1, 10, d_model }), .f32, 47);
    const attention_output = try attention.forward(attention_input);
    
    log.info("Multi-head attention: d_model={d}, heads={d}, output={d}x{d}x{d}", .{
        d_model, num_heads,
        attention_output.shape.getDim(0), attention_output.shape.getDim(1), attention_output.shape.getDim(2),
    });
}

fn demoWorkflowOrchestration() !void {
    log.info("\n🔄 Demo 3: Workflow Orchestration", .{});
    
    const allocator = std.heap.page_allocator;
    
    // Create a simple workflow
    const flow = try nen_ml.NenMLFlow.init(allocator);
    defer flow.deinit();
    
    // Add tensor operation node
    const tensor_node = try allocator.create(nen_ml.MLNode);
    tensor_node.* = nen_ml.MLNode.init("input_tensor", .tensor_op);
    tensor_node.setShapes(
        nen_ml.Shape.init(&.{ 2, 3 }),
        nen_ml.Shape.init(&.{ 2, 3 })
    );
    try flow.addNode(tensor_node);
    
    // Add linear layer node
    const linear_node = try allocator.create(nen_ml.MLNode);
    linear_node.* = nen_ml.MLNode.init("linear_layer", .linear_layer);
    linear_node.setShapes(
        nen_ml.Shape.init(&.{ 2, 3 }),
        nen_ml.Shape.init(&.{ 2, 4 })
    );
    linear_node.setLayerConfig(nen_ml.LayerConfig.init(.linear, 3, 4));
    try flow.addNode(linear_node);
    
    // Add activation node
    const activation_node = try allocator.create(nen_ml.MLNode);
    activation_node.* = nen_ml.MLNode.init("activation", .activation);
    activation_node.setShapes(
        nen_ml.Shape.init(&.{ 2, 4 }),
        nen_ml.Shape.init(&.{ 2, 4 })
    );
    try flow.addNode(activation_node);
    
    // Execute workflow
    try flow.execute();
    
    // Get statistics
    const stats = flow.getStats();
    log.info("Workflow execution: {d} nodes, {d} completed, {d} failed", .{
        stats.total_nodes,
        stats.completed_nodes,
        stats.failed_nodes,
    });
    
    log.info("Success rate: {d:.2}%", .{ stats.getSuccessRate() * 100.0 });
    log.info("Average execution time: {d:.2}μs", .{ stats.getAverageExecutionTime() / std.time.ns_per_us });
    
    // Create pre-built models
    const linear_model = try nen_ml.createLinearModel(allocator, 784, 128, 10);
    defer linear_model.deinit();
    
    log.info("Linear model created: {d} nodes", .{ linear_model.nodes.items.len });
    
    const transformer_block = try nen_ml.createTransformerBlock(allocator, 512, 8);
    defer transformer_block.deinit();
    
    log.info("Transformer block created: {d} nodes", .{ transformer_block.nodes.items.len });
}

fn demoPerformanceBenchmark(allocator: std.mem.Allocator) !void {
    log.info("\n⚡ Demo 4: Performance Benchmark", .{});
    
    const iterations = 1000;
    
    // Tensor operations benchmark
    const tensor_shape = nen_ml.Shape.init(&.{ 100, 100 });
    const tensor_a = nen_ml.Tensor.random(tensor_shape, .f32, 48);
    const tensor_b = nen_ml.Tensor.random(tensor_shape, .f32, 49);
    
    var timer = try std.time.Timer.start();
    
    for (0..iterations) |i| {
        _ = try tensor_a.add(tensor_b);
        if (i % 100 == 0) {
            log.info("Tensor operations: {d}/1000", .{i});
        }
    }
    
    const tensor_time = timer.read();
    const tensor_ops_per_sec = (iterations * std.time.ns_per_s) / tensor_time;
    
    log.info("Tensor operations: {d} ops/sec", .{ tensor_ops_per_sec });
    
    // Neural network benchmark
    const input_shape = nen_ml.Shape.init(&.{ 32, 64 });
    const input = nen_ml.Tensor.random(input_shape, .f32, 50);
    const linear = nen_ml.Linear.init(64, 128, .f32);
    
    timer.reset();
    
    for (0..iterations) |i| {
        _ = try linear.forward(input);
        if (i % 100 == 0) {
            log.info("Neural network: {d}/1000", .{i});
        }
    }
    
    const nn_time = timer.read();
    const nn_ops_per_sec = (iterations * std.time.ns_per_s) / nn_time;
    
    log.info("Neural network: {d} ops/sec", .{ nn_ops_per_sec });
    
    // Workflow benchmark
    const flow = try nen_ml.createLinearModel(allocator, 64, 128, 10);
    defer flow.deinit();
    
    timer.reset();
    
    for (0..iterations) |i| {
        try flow.execute();
        if (i % 100 == 0) {
            log.info("Workflow: {d}/1000", .{i});
        }
    }
    
    const workflow_time = timer.read();
    const workflow_ops_per_sec = (iterations * std.time.ns_per_s) / workflow_time;
    
    log.info("Workflow: {d} ops/sec", .{ workflow_ops_per_sec });
    
    // Summary
    log.info("\n📊 Performance Summary:", .{});
    log.info("Tensor operations: {d} ops/sec", .{ tensor_ops_per_sec });
    log.info("Neural network: {d} ops/sec", .{ nn_ops_per_sec });
    log.info("Workflow: {d} ops/sec", .{ workflow_ops_per_sec });
}
