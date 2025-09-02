// NenML Workflow Demo
// Demonstrates workflow orchestration with ML nodes

const std = @import("std");
const nen_ml = @import("nen_ml");

const log = std.log.scoped(.workflow_demo);

pub fn main() !void {
    log.info("🔄 NenML Workflow Demo", .{});
    
    const allocator = std.heap.page_allocator;
    
    // Create a simple workflow
    const flow = try nen_ml.NenMLFlow.init(allocator);
    defer flow.deinit();
    
    log.info("Created workflow", .{});
    
    // Add tensor operation node
    const tensor_node = try allocator.create(nen_ml.MLNode);
    tensor_node.* = nen_ml.MLNode.init("input_tensor", .tensor_op);
    tensor_node.setShapes(
        nen_ml.Shape.init(&.{ 2, 3 }),
        nen_ml.Shape.init(&.{ 2, 3 })
    );
    try flow.addNode(tensor_node);
    
    log.info("Added tensor operation node", .{});
    
    // Add linear layer node
    const linear_node = try allocator.create(nen_ml.MLNode);
    linear_node.* = nen_ml.MLNode.init("linear_layer", .linear_layer);
    linear_node.setShapes(
        nen_ml.Shape.init(&.{ 2, 3 }),
        nen_ml.Shape.init(&.{ 2, 4 })
    );
    linear_node.setLayerConfig(nen_ml.LayerConfig.init(.linear, 3, 4));
    try flow.addNode(linear_node);
    
    log.info("Added linear layer node", .{});
    
    // Add activation node
    const activation_node = try allocator.create(nen_ml.MLNode);
    activation_node.* = nen_ml.MLNode.init("activation", .activation);
    activation_node.setShapes(
        nen_ml.Shape.init(&.{ 2, 4 }),
        nen_ml.Shape.init(&.{ 2, 4 })
    );
    try flow.addNode(activation_node);
    
    log.info("Added activation node", .{});
    
    // Add layer normalization node
    const norm_node = try allocator.create(nen_ml.MLNode);
    norm_node.* = nen_ml.MLNode.init("layer_norm", .layer_norm);
    norm_node.setShapes(
        nen_ml.Shape.init(&.{ 2, 4 }),
        nen_ml.Shape.init(&.{ 2, 4 })
    );
    try flow.addNode(norm_node);
    
    log.info("Added layer normalization node", .{});
    
    // Execute workflow
    log.info("Executing workflow...", .{});
    try flow.execute();
    
    // Get statistics
    const stats = flow.getStats();
    log.info("Workflow execution completed:", .{});
    log.info("  Total nodes: {d}", .{ stats.total_nodes });
    log.info("  Completed nodes: {d}", .{ stats.completed_nodes });
    log.info("  Failed nodes: {d}", .{ stats.failed_nodes });
    log.info("  Success rate: {d:.2}%", .{ stats.getSuccessRate() * 100.0 });
    log.info("  Average execution time: {d:.2}μs", .{ stats.getAverageExecutionTime() / std.time.ns_per_us });
    
    // Create pre-built models
    log.info("\n🏗️ Creating pre-built models:", .{});
    
    const linear_model = try nen_ml.createLinearModel(allocator, 784, 128, 10);
    defer linear_model.deinit();
    
    log.info("Linear model: {d} nodes", .{ linear_model.nodes.items.len });
    
    const transformer_block = try nen_ml.createTransformerBlock(allocator, 512, 8);
    defer transformer_block.deinit();
    
    log.info("Transformer block: {d} nodes", .{ transformer_block.nodes.items.len });
    
    // Execute pre-built models
    log.info("\n🚀 Executing pre-built models:", .{});
    
    try linear_model.execute();
    const linear_stats = linear_model.getStats();
    log.info("Linear model: {d} nodes, {d:.2}% success", .{
        linear_stats.total_nodes,
        linear_stats.getSuccessRate() * 100.0,
    });
    
    try transformer_block.execute();
    const transformer_stats = transformer_block.getStats();
    log.info("Transformer block: {d} nodes, {d:.2}% success", .{
        transformer_stats.total_nodes,
        transformer_stats.getSuccessRate() * 100.0,
    });
    
    // Performance benchmark
    log.info("\n⚡ Performance benchmark:", .{});
    
    const iterations = 100;
    var timer = try std.time.Timer.start();
    
    for (0..iterations) |i| {
        try flow.execute();
        if (i % 20 == 0) {
            log.info("  Progress: {d}/100", .{i});
        }
    }
    
    const total_time = timer.read();
    const avg_time = total_time / iterations;
    const ops_per_sec = (iterations * std.time.ns_per_s) / total_time;
    
    log.info("  Total time: {d:.2}ms", .{ @as(f64, @floatFromInt(total_time)) / std.time.ns_per_ms });
    log.info("  Average time: {d:.2}μs", .{ @as(f64, @floatFromInt(avg_time)) / std.time.ns_per_us });
    log.info("  Operations per second: {d}", .{ ops_per_sec });
    
    log.info("✅ Workflow demo completed!", .{});
}
