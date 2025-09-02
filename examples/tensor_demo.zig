// NenML Tensor Demo
// Demonstrates tensor operations with static memory

const std = @import("std");
const nen_ml = @import("nen_ml");

const log = std.log.scoped(.tensor_demo);

pub fn main() !void {
    log.info("📊 NenML Tensor Demo", .{});
    
    // Create tensors
    const shape = nen_ml.Shape.init(&.{ 3, 4 });
    const tensor = nen_ml.Tensor.random(shape, .f32, 42);
    
    log.info("Created tensor: {d}x{d}", .{
        tensor.shape.getDim(0),
        tensor.shape.getDim(1),
    });
    
    // Basic operations
    const zeros = nen_ml.Tensor.zeros(shape, .f32);
    const ones = nen_ml.Tensor.ones(shape, .f32);
    
    const sum = try tensor.add(ones);
    const product = try tensor.multiply(ones);
    
    log.info("Operations completed successfully", .{});
    
    // Matrix multiplication
    const matrix_a = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 3, 2 }), .f32, 43);
    const matrix_b = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 2, 4 }), .f32, 44);
    
    const matmul_result = try matrix_a.matmul(matrix_b);
    
    log.info("Matrix multiplication: {d}x{d} @ {d}x{d} = {d}x{d}", .{
        matrix_a.shape.getDim(0), matrix_a.shape.getDim(1),
        matrix_b.shape.getDim(0), matrix_b.shape.getDim(1),
        matmul_result.shape.getDim(0), matmul_result.shape.getDim(1),
    });
    
    // Activation functions
    const relu_result = tensor.relu();
    const sigmoid_result = tensor.sigmoid();
    
    log.info("Activation functions applied", .{});
    
    // Memory pool demo
    var pool = nen_ml.TensorPool.init();
    const pool_tensor = pool.allocate(shape, .f32);
    
    if (pool_tensor) |tensor_ptr| {
        log.info("Allocated tensor from pool", .{});
        pool.deallocate(tensor_ptr);
        log.info("Deallocated tensor back to pool", .{});
    }
    
    log.info("Available tensors in pool: {d}", .{ pool.getAvailableCount() });
    
    log.info("✅ Tensor demo completed!", .{});
}
