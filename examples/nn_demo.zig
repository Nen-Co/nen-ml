// NenML Neural Network Demo
// Demonstrates neural network layers with static memory

const std = @import("std");
const nen_ml = @import("nen_ml");

const log = std.log.scoped(.nn_demo);

pub fn main() !void {
    log.info("🧠 NenML Neural Network Demo", .{});
    
    // Linear layer
    const input_shape = nen_ml.Shape.init(&.{ 2, 3 });
    const input = nen_ml.Tensor.random(input_shape, .f32, 42);
    
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
    
    const relu_output = relu.forward(linear_output);
    const gelu_output = gelu.forward(linear_output);
    const silu_output = silu.forward(linear_output);
    
    log.info("Activation functions applied: relu, gelu, silu", .{});
    
    // Layer normalization
    const norm_shape = nen_ml.Shape.init(&.{ 4 });
    const layer_norm = nen_ml.LayerNorm.init(norm_shape, .f32);
    const norm_output = layer_norm.forward(linear_output);
    
    log.info("Layer normalization applied", .{});
    
    // Dropout
    const dropout = nen_ml.Dropout.init(0.1);
    const dropout_output = dropout.forward(norm_output);
    
    log.info("Dropout applied: rate=0.1", .{});
    
    // Token embedding
    const vocab_size: usize = 1000;
    const embedding_dim: usize = 64;
    const embedding = nen_ml.TokenEmbedding.init(vocab_size, embedding_dim, .f32);
    
    const token_shape = nen_ml.Shape.init(&.{ 1, 5 }); // batch_size=1, seq_len=5
    const tokens = nen_ml.Tensor.random(token_shape, .i32, 43);
    const embedded = try embedding.forward(tokens);
    
    log.info("Token embedding: vocab={d}, dim={d}, output={d}x{d}x{d}", .{
        vocab_size, embedding_dim,
        embedded.shape.getDim(0), embedded.shape.getDim(1), embedded.shape.getDim(2),
    });
    
    // Multi-head attention
    const d_model: usize = 64;
    const num_heads: usize = 8;
    const attention = nen_ml.MultiHeadAttention.init(d_model, num_heads, .f32);
    
    const attention_input = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 1, 10, d_model }), .f32, 44);
    const attention_output = try attention.forward(attention_input);
    
    log.info("Multi-head attention: d_model={d}, heads={d}, output={d}x{d}x{d}", .{
        d_model, num_heads,
        attention_output.shape.getDim(0), attention_output.shape.getDim(1), attention_output.shape.getDim(2),
    });
    
    // Complete neural network pipeline
    log.info("\n🔄 Complete Neural Network Pipeline:", .{});
    
    // Input -> Linear -> ReLU -> LayerNorm -> Dropout
    const pipeline_input = nen_ml.Tensor.random(nen_ml.Shape.init(&.{ 1, 10 }), .f32, 45);
    
    const pipeline_linear = nen_ml.Linear.init(10, 20, .f32);
    const pipeline_relu = nen_ml.Activation.relu;
    const pipeline_norm = nen_ml.LayerNorm.init(nen_ml.Shape.init(&.{ 20 }), .f32);
    const pipeline_dropout = nen_ml.Dropout.init(0.1);
    
    const step1 = try pipeline_linear.forward(pipeline_input);
    const step2 = pipeline_relu.forward(step1);
    const step3 = pipeline_norm.forward(step2);
    const step4 = pipeline_dropout.forward(step3);
    
    log.info("Pipeline: {d}x{d} -> {d}x{d} -> {d}x{d} -> {d}x{d} -> {d}x{d}", .{
        pipeline_input.shape.getDim(0), pipeline_input.shape.getDim(1),
        step1.shape.getDim(0), step1.shape.getDim(1),
        step2.shape.getDim(0), step2.shape.getDim(1),
        step3.shape.getDim(0), step3.shape.getDim(1),
        step4.shape.getDim(0), step4.shape.getDim(1),
    });
    
    log.info("✅ Neural network demo completed!", .{});
}
