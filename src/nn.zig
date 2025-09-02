// NenNN: Neural Network layers built from scratch following Nen principles
// Static memory, zero-allocation, inline functions, static typing

const std = @import("std");
const nentensor = @import("tensor.zig");

// Core neural network layer types
pub const LayerType = enum {
    linear,
    conv2d,
    maxpool,
    dropout,
    batch_norm,
    layer_norm,
    embedding,
    attention,
};

// Linear layer with static memory
pub const Linear = struct {
    weight: nentensor.Tensor,
    bias: ?nentensor.Tensor,
    
    pub inline fn init(input_size: usize, output_size: usize, dtype: nentensor.DataType) Linear {
        const weight_shape = nentensor.Shape.init(&.{ input_size, output_size });
        const bias_shape = nentensor.Shape.init(&.{ output_size });
        
        const weight = nentensor.Tensor.random(weight_shape, dtype, 42);
        const bias = nentensor.Tensor.random(bias_shape, dtype, 43);
        
        return Linear{
            .weight = weight,
            .bias = bias,
        };
    }
    
    pub inline fn forward(self: Linear, input: nentensor.Tensor) error{InvalidInput}!nentensor.Tensor {
        // Matrix multiplication: input @ weight
        const output = try input.matmul(self.weight);
        
        // Add bias if present
        if (self.bias) |bias| {
            // Broadcast bias to match output shape
            return try self.broadcastAdd(output, bias);
        }
        
        return output;
    }
    
    inline fn broadcastAdd(self: Linear, output: nentensor.Tensor, bias: nentensor.Tensor) error{InvalidInput}!nentensor.Tensor {
        _ = self;
        // Simple broadcasting for now - assumes output is 2D and bias is 1D
        if (output.shape.rank != 2 or bias.shape.rank != 1) {
            return error.InvalidInput;
        }
        
        var result = nentensor.Tensor.init(output.shape, output.dtype);
        const batch_size = output.shape.getDim(0);
        const output_size = output.shape.getDim(1);
        
        for (0..batch_size) |i| {
            for (0..output_size) |j| {
                const output_idx = i * output_size + j;
                const bias_idx = j;
                
                const output_val = std.mem.readIntLittle(f32, output.data[output_idx * 4..][0..4]);
                const bias_val = std.mem.readIntLittle(f32, bias.data[bias_idx * 4..][0..4]);
                const sum = output_val + bias_val;
                
                std.mem.writeIntLittle(f32, result.data[output_idx * 4..][0..4], sum);
            }
        }
        
        result.data_len = output.data_len;
        return result;
    }
};

// Activation functions
pub const Activation = union(enum) {
    relu,
    sigmoid,
    tanh,
    gelu,
    silu,
    leaky_relu: f32,
    elu: f32,
    
    pub inline fn forward(self: Activation, input: nentensor.Tensor) nentensor.Tensor {
        return switch (self) {
            .relu => input.relu(),
            .sigmoid => input.sigmoid(),
            .tanh => self.tanh_forward(input),
            .gelu => self.gelu_forward(input),
            .silu => self.silu_forward(input),
            .leaky_relu => |slope| self.leaky_relu_forward(input, slope),
            .elu => |alpha| self.elu_forward(input, alpha),
        };
    }
    
    inline fn tanh_forward(self: Activation, input: nentensor.Tensor) nentensor.Tensor {
        _ = self;
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const tanh_x = std.math.tanh(x);
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], tanh_x);
        }
        
        result.data_len = input.data_len;
        return result;
    }
    
    inline fn gelu_forward(self: Activation, input: nentensor.Tensor) nentensor.Tensor {
        _ = self;
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const gelu_x = 0.5 * x * (1.0 + std.math.tanh(std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * x * x * x)));
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], gelu_x);
        }
        
        result.data_len = input.data_len;
        return result;
    }
    
    inline fn silu_forward(self: Activation, input: nentensor.Tensor) nentensor.Tensor {
        _ = self;
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const silu_x = x / (1.0 + std.math.exp(-x));
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], silu_x);
        }
        
        result.data_len = input.data_len;
        return result;
    }
    
    inline fn leaky_relu_forward(self: Activation, input: nentensor.Tensor, slope: f32) nentensor.Tensor {
        _ = self;
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const leaky_x = if (x > 0) x else slope * x;
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], leaky_x);
        }
        
        result.data_len = input.data_len;
        return result;
    }
    
    inline fn elu_forward(self: Activation, input: nentensor.Tensor, alpha: f32) nentensor.Tensor {
        _ = self;
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const elu_x = if (x > 0) x else alpha * (std.math.exp(x) - 1.0);
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], elu_x);
        }
        
        result.data_len = input.data_len;
        return result;
    }
};

// Layer normalization
pub const LayerNorm = struct {
    weight: nentensor.Tensor,
    bias: ?nentensor.Tensor,
    eps: f32,
    
    pub inline fn init(normalized_shape: nentensor.Shape, dtype: nentensor.DataType) LayerNorm {
        const weight = nentensor.Tensor.ones(normalized_shape, dtype);
        const bias = nentensor.Tensor.zeros(normalized_shape, dtype);
        
        return LayerNorm{
            .weight = weight,
            .bias = bias,
            .eps = 1e-5,
        };
    }
    
    pub inline fn forward(self: LayerNorm, input: nentensor.Tensor) nentensor.Tensor {
        // Calculate mean and variance
        const mean = self.calculateMean(input);
        const variance = self.calculateVariance(input, mean);
        
        // Normalize
        var normalized = self.normalize(input, mean, variance);
        
        // Apply weight and bias
        normalized = self.applyWeightAndBias(normalized);
        
        return normalized;
    }
    
    inline fn calculateMean(self: LayerNorm, input: nentensor.Tensor) f32 {
        _ = self;
        const elements = input.totalElements();
        var sum: f32 = 0.0;
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            sum += x;
        }
        
        return sum / @as(f32, @floatFromInt(elements));
    }
    
    inline fn calculateVariance(self: LayerNorm, input: nentensor.Tensor, mean: f32) f32 {
        _ = self;
        const elements = input.totalElements();
        var sum_sq: f32 = 0.0;
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const diff = x - mean;
            sum_sq += diff * diff;
        }
        
        return sum_sq / @as(f32, @floatFromInt(elements));
    }
    
    inline fn normalize(self: LayerNorm, input: nentensor.Tensor, mean: f32, variance: f32) nentensor.Tensor {
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        const std_dev = std.math.sqrt(variance + self.eps);
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const normalized = (x - mean) / std_dev;
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], normalized);
        }
        
        result.data_len = input.data_len;
        return result;
    }
    
    inline fn applyWeightAndBias(self: LayerNorm, normalized: nentensor.Tensor) nentensor.Tensor {
        var result = nentensor.Tensor.init(normalized.shape, normalized.dtype);
        const elements = normalized.totalElements();
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, normalized.data[offset..][0..4]);
            const weight_val = std.mem.readIntLittle(f32, self.weight.data[offset..][0..4]);
            var output = x * weight_val;
            
            if (self.bias) |bias| {
                const bias_val = std.mem.readIntLittle(f32, bias.data[offset..][0..4]);
                output += bias_val;
            }
            
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], output);
        }
        
        result.data_len = normalized.data_len;
        return result;
    }
};

// Dropout layer
pub const Dropout = struct {
    rate: f32,
    training: bool,
    
    pub inline fn init(rate: f32) Dropout {
        return Dropout{
            .rate = rate,
            .training = true,
        };
    }
    
    pub inline fn forward(self: Dropout, input: nentensor.Tensor) nentensor.Tensor {
        if (!self.training or self.rate == 0.0) {
            return input;
        }
        
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        const scale = 1.0 / (1.0 - self.rate);
        
        var rng = std.rand.Xoshiro256.init(42);
        
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            
            const random_val = rng.random().float(f32);
            const mask = if (random_val > self.rate) 1.0 else 0.0;
            const output = x * mask * scale;
            
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], output);
        }
        
        result.data_len = input.data_len;
        return result;
    }
    
    pub inline fn setTraining(self: *Dropout, training: bool) void {
        self.training = training;
    }
};

// Token embedding layer
pub const TokenEmbedding = struct {
    weight: nentensor.Tensor,
    
    pub inline fn init(vocab_size: usize, embedding_dim: usize, dtype: nentensor.DataType) TokenEmbedding {
        const weight_shape = nentensor.Shape.init(&.{ vocab_size, embedding_dim });
        const weight = nentensor.Tensor.random(weight_shape, dtype, 44);
        
        return TokenEmbedding{
            .weight = weight,
        };
    }
    
    pub inline fn forward(self: TokenEmbedding, indices: nentensor.Tensor) error{InvalidInput}!nentensor.Tensor {
        // Simple embedding lookup
        if (indices.dtype != .i32 and indices.dtype != .u8) {
            return error.InvalidInput;
        }
        
        const batch_size = indices.shape.getDim(0);
        const seq_len = indices.shape.getDim(1);
        const embedding_dim = self.weight.shape.getDim(1);
        
        const output_shape = nentensor.Shape.init(&.{ batch_size, seq_len, embedding_dim });
        var result = nentensor.Tensor.init(output_shape, self.weight.dtype);
        
        for (0..batch_size) |b| {
            for (0..seq_len) |s| {
                const idx = b * seq_len + s;
                const token_id = if (indices.dtype == .i32) 
                    std.mem.readIntLittle(i32, indices.data[idx * 4..][0..4])
                else
                    indices.data[idx];
                
                const token_id_usize = @as(usize, @intCast(@abs(token_id)));
                
                // Copy embedding vector
                for (0..embedding_dim) |d| {
                    const src_offset = (token_id_usize * embedding_dim + d) * 4;
                    const dst_offset = (b * seq_len * embedding_dim + s * embedding_dim + d) * 4;
                    
                    const embedding_val = std.mem.readIntLittle(f32, self.weight.data[src_offset..][0..4]);
                    std.mem.writeIntLittle(f32, result.data[dst_offset..][0..4], embedding_val);
                }
            }
        }
        
        result.data_len = batch_size * seq_len * embedding_dim * 4;
        return result;
    }
};

// Multi-head attention (simplified)
pub const MultiHeadAttention = struct {
    query_weight: nentensor.Tensor,
    key_weight: nentensor.Tensor,
    value_weight: nentensor.Tensor,
    output_weight: nentensor.Tensor,
    num_heads: usize,
    head_dim: usize,
    
    pub inline fn init(d_model: usize, num_heads: usize, dtype: nentensor.DataType) MultiHeadAttention {
        const head_dim = d_model / num_heads;
        const weight_shape = nentensor.Shape.init(&.{ d_model, d_model });
        
        const query_weight = nentensor.Tensor.random(weight_shape, dtype, 45);
        const key_weight = nentensor.Tensor.random(weight_shape, dtype, 46);
        const value_weight = nentensor.Tensor.random(weight_shape, dtype, 47);
        const output_weight = nentensor.Tensor.random(weight_shape, dtype, 48);
        
        return MultiHeadAttention{
            .query_weight = query_weight,
            .key_weight = key_weight,
            .value_weight = value_weight,
            .output_weight = output_weight,
            .num_heads = num_heads,
            .head_dim = head_dim,
        };
    }
    
    pub inline fn forward(self: MultiHeadAttention, input: nentensor.Tensor) error{InvalidInput}!nentensor.Tensor {
        // Simplified attention - just matrix multiplications
        const query = try input.matmul(self.query_weight);
        const key = try input.matmul(self.key_weight);
        const value = try input.matmul(self.value_weight);
        
        // Simple attention scores
        const attention_scores = try query.matmul(key);
        
        // Apply softmax (simplified)
        const attention_weights = self.softmax(attention_scores);
        
        // Apply attention to values
        const attended = try attention_weights.matmul(value);
        
        // Final output projection
        return try attended.matmul(self.output_weight);
    }
    
    inline fn softmax(self: MultiHeadAttention, input: nentensor.Tensor) nentensor.Tensor {
        _ = self;
        var result = nentensor.Tensor.init(input.shape, input.dtype);
        const elements = input.totalElements();
        
        // Find max for numerical stability
        var max_val: f32 = -std.math.f32_max;
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            if (x > max_val) max_val = x;
        }
        
        // Calculate exp and sum
        var sum: f32 = 0.0;
        for (0..elements) |i| {
            const offset = i * 4;
            const x = std.mem.readIntLittle(f32, input.data[offset..][0..4]);
            const exp_x = std.math.exp(x - max_val);
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], exp_x);
            sum += exp_x;
        }
        
        // Normalize
        for (0..elements) |i| {
            const offset = i * 4;
            const exp_x = std.mem.readIntLittle(f32, result.data[offset..][0..4]);
            const softmax_x = exp_x / sum;
            std.mem.writeIntLittle(f32, result.data[offset..][0..4], softmax_x);
        }
        
        result.data_len = input.data_len;
        return result;
    }
};

// Test functions
test "linear layer" {
    const input_shape = nentensor.Shape.init(&.{ 2, 3 });
    const input = nentensor.Tensor.random(input_shape, .f32, 42);
    
    const linear = nentensor.Linear.init(3, 4, .f32);
    const output = try linear.forward(input);
    
    try std.testing.expectEqual(@as(u8, 2), output.shape.rank);
    try std.testing.expectEqual(@as(usize, 2), output.shape.getDim(0));
    try std.testing.expectEqual(@as(usize, 4), output.shape.getDim(1));
}

test "activation functions" {
    const shape = nentensor.Shape.init(&.{ 2, 2 });
    const input = nentensor.Tensor.random(shape, .f32, 42);
    
    const relu = nentensor.Activation.relu;
    const output = relu.forward(input);
    
    try std.testing.expectEqual(input.shape.rank, output.shape.rank);
    try std.testing.expectEqual(input.totalElements(), output.totalElements());
}

test "layer normalization" {
    const shape = nentensor.Shape.init(&.{ 2, 3 });
    const input = nentensor.Tensor.random(shape, .f32, 42);
    
    const norm_shape = nentensor.Shape.init(&.{ 3 });
    const layer_norm = nentensor.LayerNorm.init(norm_shape, .f32);
    const output = layer_norm.forward(input);
    
    try std.testing.expectEqual(input.shape.rank, output.shape.rank);
    try std.testing.expectEqual(input.totalElements(), output.totalElements());
}
