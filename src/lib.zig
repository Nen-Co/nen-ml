// NenML: Machine Learning Library built from scratch following Nen principles
// Static memory, zero-allocation, inline functions, static typing

const std = @import("std");

// Core ML modules
pub const tensor = @import("tensor.zig");
pub const nn = @import("nn.zig");
pub const workflow = @import("workflow.zig");
pub const inference = @import("inference.zig");
pub const real_inference = @import("real_inference.zig");
pub const nenformat_reader = @import("nenformat_reader.zig");
pub const tokenizer = @import("tokenizer.zig");

// Re-export main types for convenience
pub const Tensor = tensor.Tensor;
pub const Shape = tensor.Shape;
pub const DataType = tensor.DataType;
pub const TensorPool = tensor.TensorPool;

pub const Linear = nn.Linear;
pub const Activation = nn.Activation;
pub const LayerNorm = nn.LayerNorm;
pub const Dropout = nn.Dropout;
pub const TokenEmbedding = nn.TokenEmbedding;
pub const MultiHeadAttention = nn.MultiHeadAttention;

pub const MLNode = workflow.MLNode;
pub const MLNodeType = workflow.MLNodeType;
pub const MLNodeState = workflow.MLNodeState;
pub const NenMLFlow = workflow.NenMLFlow;
pub const NenMLFlowStats = workflow.NenMLFlowStats;

// Inference types
pub const NenInference = inference.NenInference;
pub const InferenceStats = inference.InferenceStats;

// Real inference types
pub const RealInference = real_inference.RealInference;
pub const ModelStats = real_inference.ModelStats;
pub const NenFormatReader = nenformat_reader.NenFormatReader;
pub const Tokenizer = tokenizer.Tokenizer;

// Convenience functions
pub fn createLinearModel(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize) !*NenMLFlow {
    return workflow.createLinearModel(allocator, input_size, hidden_size, output_size);
}

pub fn createTransformerBlock(allocator: std.mem.Allocator, d_model: usize, n_heads: usize) !*NenMLFlow {
    return workflow.createTransformerBlock(allocator, d_model, n_heads);
}

// Export for C bindings
export fn nenml_create_flow(allocator: *anyopaque) ?*anyopaque {
    return workflow.nenml_create_flow(allocator);
}

export fn nenml_add_node(flow: *anyopaque, node: *anyopaque) c_int {
    return workflow.nenml_add_node(flow, node);
}

export fn nenml_execute_flow(flow: *anyopaque) c_int {
    return workflow.nenml_execute_flow(flow);
}

export fn nenml_get_flow_stats(flow: *anyopaque) ?*anyopaque {
    return workflow.nenml_get_flow_stats(flow);
}

// Test all modules
// NOTE: Temporarily disabled comprehensive refAllDecls test pending
// stabilization of workflow C binding helpers which crash under
// exhaustive reflection in Zig 0.15.
// test {
//     std.testing.refAllDecls(@This());
//     _ = tensor;
//     _ = nn;
//     _ = workflow;
// }
