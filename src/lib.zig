// NenML: Machine Learning Library built from scratch following Nen principles
// Static memory, zero-allocation, inline functions, static typing

const std = @import("std");

// Core ML modules
pub const tensor = @import("tensor.zig");
pub const nn = @import("nn.zig");
pub const workflow = @import("workflow.zig");

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

// Convenience functions
pub fn createLinearModel(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize) !*NenMLFlow {
    return workflow.createLinearModel(allocator, input_size, hidden_size, output_size);
}

pub fn createTransformerBlock(allocator: std.mem.Allocator, d_model: usize, n_heads: usize) !*NenMLFlow {
    return workflow.createTransformerBlock(allocator, d_model, n_heads);
}

// Export for C bindings
export fn nenml_create_flow(allocator: *anyopaque) *anyopaque {
    return workflow.nenml_create_flow(allocator);
}

export fn nenml_add_node(flow: *anyopaque, node: *anyopaque) c_int {
    return workflow.nenml_add_node(flow, node);
}

export fn nenml_execute_flow(flow: *anyopaque) c_int {
    return workflow.nenml_execute_flow(flow);
}

export fn nenml_get_flow_stats(flow: *anyopaque) *anyopaque {
    return workflow.nenml_get_flow_stats(flow);
}

// Test all modules
test {
    std.testing.refAllDecls(@This());
    
    // Test tensor module
    _ = tensor;
    
    // Test neural network module
    _ = nn;
    
    // Test workflow module
    _ = workflow;
}
