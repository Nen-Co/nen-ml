// NenTensor: Tensor operations built from scratch following Nen principles
// Static memory, zero-allocation, inline functions, static typing

const std = @import("std");

// Core tensor types - using static memory pools
pub const DataType = enum {
    f32,
    f16,
    i32,
    i64,
    u8,
    bool,

    pub inline fn size(self: DataType) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
            .i32 => 4,
            .i64 => 8,
            .u8 => 1,
            .bool => 1,
        };
    }

    pub inline fn isFloat(self: DataType) bool {
        return switch (self) {
            .f32, .f16 => true,
            else => false,
        };
    }

    pub inline fn isInteger(self: DataType) bool {
        return switch (self) {
            .i32, .i64, .u8 => true,
            else => false,
        };
    }
};

// Static shape with compile-time size limits
pub const Shape = struct {
    dims: [8]usize, // Max 8 dimensions
    rank: u8,

    pub inline fn init(dims: []const usize) Shape {
        var shape = Shape{ .dims = undefined, .rank = 0 };
        const rank = @min(dims.len, 8);
        shape.rank = @intCast(rank);
        for (0..rank) |i| {
            shape.dims[i] = dims[i];
        }
        return shape;
    }

    pub inline fn getRank(self: Shape) u8 {
        return self.rank;
    }

    pub inline fn totalElements(self: Shape) usize {
        var total: usize = 1;
        for (0..self.rank) |i| {
            total *= self.dims[i];
        }
        return total;
    }

    pub inline fn getDim(self: Shape, index: u8) usize {
        if (index < self.rank) {
            return self.dims[index];
        }
        return 0;
    }

    pub inline fn isCompatible(self: Shape, other: Shape) bool {
        if (self.rank != other.rank) return false;
        for (0..self.rank) |i| {
            if (self.dims[i] != other.dims[i]) return false;
        }
        return true;
    }
};

// Static tensor with fixed memory pool
pub const Tensor = struct {
    data: [1024]u8, // 1KB static buffer
    data_len: usize,
    shape: Shape,
    dtype: DataType,

    pub inline fn init(shape: Shape, dtype: DataType) Tensor {
        return Tensor{
            .data = undefined,
            .data_len = 0,
            .shape = shape,
            .dtype = dtype,
        };
    }

    pub inline fn setData(self: *Tensor, new_data: []const u8) error{DataTooLarge}!void {
        const required_size = self.shape.totalElements() * self.dtype.size();
        if (required_size > self.data.len) {
            return error.DataTooLarge;
        }
        if (new_data.len > self.data.len) {
            return error.DataTooLarge;
        }
        @memcpy(self.data[0..new_data.len], new_data);
        self.data_len = new_data.len;
    }

    pub inline fn getData(self: Tensor) []const u8 {
        return self.data[0..self.data_len];
    }

    pub inline fn getShape(self: Tensor) Shape {
        return self.shape;
    }

    pub inline fn getDtype(self: Tensor) DataType {
        return self.dtype;
    }

    pub inline fn totalElements(self: Tensor) usize {
        return self.shape.totalElements();
    }

    pub inline fn size(self: Tensor) usize {
        return self.totalElements() * self.dtype.size();
    }

    // Element-wise operations
    pub inline fn add(self: Tensor, other: Tensor) error{ IncompatibleShapes, IncompatibleTypes, UnsupportedOperation }!Tensor {
        if (!self.shape.isCompatible(other.shape)) {
            return error.IncompatibleShapes;
        }
        if (self.dtype != other.dtype) {
            return error.IncompatibleTypes;
        }

        var result = Tensor.init(self.shape, self.dtype);
        const data_size = self.dtype.size();
        const elements = self.totalElements();

        for (0..elements) |i| {
            const offset = i * data_size;
            switch (self.dtype) {
                .f32 => {
                    const a_bytes = self.data[offset..][0..4];
                    const b_bytes = other.data[offset..][0..4];
                    const a: f32 = @bitCast(std.mem.bytesAsValue(u32, a_bytes).*);
                    const b: f32 = @bitCast(std.mem.bytesAsValue(u32, b_bytes).*);
                    const sum: f32 = a + b;
                    const sum_u32: u32 = @bitCast(sum);
                    std.mem.bytesAsValue(u32, result.data[offset..][0..4]).* = sum_u32;
                },
                .i32 => {
                    const a = std.mem.bytesAsValue(i32, self.data[offset..][0..4]).*;
                    const b = std.mem.bytesAsValue(i32, other.data[offset..][0..4]).*;
                    const sum: i32 = a + b;
                    std.mem.bytesAsValue(i32, result.data[offset..][0..4]).* = sum;
                },
                else => return error.UnsupportedOperation,
            }
        }

        result.data_len = self.data_len;
        return result;
    }

    pub inline fn multiply(self: Tensor, other: Tensor) error{ IncompatibleShapes, IncompatibleTypes, UnsupportedOperation }!Tensor {
        if (!self.shape.isCompatible(other.shape)) {
            return error.IncompatibleShapes;
        }
        if (self.dtype != other.dtype) {
            return error.IncompatibleTypes;
        }

        var result = Tensor.init(self.shape, self.dtype);
        const data_size = self.dtype.size();
        const elements = self.totalElements();

        for (0..elements) |i| {
            const offset = i * data_size;
            switch (self.dtype) {
                .f32 => {
                    const a_bytes = self.data[offset..][0..4];
                    const b_bytes = other.data[offset..][0..4];
                    const a: f32 = @bitCast(std.mem.bytesAsValue(u32, a_bytes).*);
                    const b: f32 = @bitCast(std.mem.bytesAsValue(u32, b_bytes).*);
                    const product: f32 = a * b;
                    const product_u32: u32 = @bitCast(product);
                    std.mem.bytesAsValue(u32, result.data[offset..][0..4]).* = product_u32;
                },
                .i32 => {
                    const a = std.mem.bytesAsValue(i32, self.data[offset..][0..4]).*;
                    const b = std.mem.bytesAsValue(i32, other.data[offset..][0..4]).*;
                    const product: i32 = a * b;
                    std.mem.bytesAsValue(i32, result.data[offset..][0..4]).* = product;
                },
                else => return error.UnsupportedOperation,
            }
        }

        result.data_len = self.data_len;
        return result;
    }

    // Matrix operations
    pub inline fn matmul(self: Tensor, other: Tensor) error{InvalidMatrixShape}!Tensor {
        if (self.shape.rank != 2 or other.shape.rank != 2) {
            return error.InvalidMatrixShape;
        }
        if (self.shape.getDim(1) != other.shape.getDim(0)) {
            return error.InvalidMatrixShape;
        }

        const m = self.shape.getDim(0);
        const k = self.shape.getDim(1);
        const n = other.shape.getDim(1);

        const result_shape = Shape.init(&.{ m, n });
        var result = Tensor.init(result_shape, self.dtype);

        // Simple matrix multiplication
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0.0;
                for (0..k) |l| {
                    const a_idx = i * k + l;
                    const b_idx = l * n + j;
                    const a_bytes = self.data[a_idx * 4 ..][0..4];
                    const b_bytes = other.data[b_idx * 4 ..][0..4];
                    const a: f32 = @bitCast(std.mem.bytesAsValue(u32, a_bytes).*);
                    const b: f32 = @bitCast(std.mem.bytesAsValue(u32, b_bytes).*);
                    sum += a * b;
                }
                const result_idx = i * n + j;
                const sum_u32: u32 = @bitCast(sum);
                std.mem.bytesAsValue(u32, result.data[result_idx * 4 ..][0..4]).* = sum_u32;
            }
        }

        result.data_len = m * n * 4;
        return result;
    }

    // Activation functions
    pub inline fn relu(self: Tensor) Tensor {
        var result = Tensor.init(self.shape, self.dtype);
        const data_size = self.dtype.size();
        const elements = self.totalElements();

        for (0..elements) |i| {
            const offset = i * data_size;
            if (self.dtype == .f32) {
                const x_bytes = self.data[offset..][0..4];
                const x: f32 = @bitCast(std.mem.bytesAsValue(u32, x_bytes).*);
                const relu_x: f32 = if (x > 0) x else 0;
                const relu_u32: u32 = @bitCast(relu_x);
                std.mem.bytesAsValue(u32, result.data[offset..][0..4]).* = relu_u32;
            }
        }

        result.data_len = self.data_len;
        return result;
    }

    pub inline fn sigmoid(self: Tensor) Tensor {
        var result = Tensor.init(self.shape, self.dtype);
        const data_size = self.dtype.size();
        const elements = self.totalElements();

        for (0..elements) |i| {
            const offset = i * data_size;
            if (self.dtype == .f32) {
                const x_bytes = self.data[offset..][0..4];
                const x: f32 = @bitCast(std.mem.bytesAsValue(u32, x_bytes).*);
                const sigmoid_x: f32 = 1.0 / (1.0 + std.math.exp(-x));
                const sig_u32: u32 = @bitCast(sigmoid_x);
                std.mem.bytesAsValue(u32, result.data[offset..][0..4]).* = sig_u32;
            }
        }

        result.data_len = self.data_len;
        return result;
    }

    // Utility functions
    pub inline fn zeros(shape: Shape, dtype: DataType) Tensor {
        var tensor = Tensor.init(shape, dtype);
        const tensor_size = shape.totalElements() * dtype.size();
        @memset(tensor.data[0..tensor_size], 0);
        tensor.data_len = tensor_size;
        return tensor;
    }

    pub inline fn ones(shape: Shape, dtype: DataType) Tensor {
        var tensor = Tensor.init(shape, dtype);
        const elements = shape.totalElements();
        const data_size = dtype.size();

        for (0..elements) |i| {
            const offset = i * data_size;
            switch (dtype) {
                .f32 => {
                    const one_u32: u32 = @bitCast(@as(f32, 1.0));
                    std.mem.bytesAsValue(u32, tensor.data[offset..][0..4]).* = one_u32;
                },
                .i32 => {
                    std.mem.bytesAsValue(i32, tensor.data[offset..][0..4]).* = 1;
                },
                else => {},
            }
        }

        tensor.data_len = elements * data_size;
        return tensor;
    }

    pub inline fn random(shape: Shape, dtype: DataType, seed: u64) Tensor {
        var tensor = Tensor.init(shape, dtype);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const elements = shape.totalElements();
        const data_size = dtype.size();

        for (0..elements) |i| {
            const offset = i * data_size;
            switch (dtype) {
                .f32 => {
                    const value = rng.float(f32);
                    const v_u32: u32 = @bitCast(value);
                    std.mem.bytesAsValue(u32, tensor.data[offset..][0..4]).* = v_u32;
                },
                .i32 => {
                    const value = rng.int(i32);
                    std.mem.bytesAsValue(i32, tensor.data[offset..][0..4]).* = value;
                },
                else => {},
            }
        }

        tensor.data_len = elements * data_size;
        return tensor;
    }
};

// Static tensor pool for efficient memory management
pub const TensorPool = struct {
    tensors: [16]Tensor, // Pool of 16 tensors
    used: [16]bool,

    pub inline fn init() TensorPool {
        return TensorPool{
            .tensors = undefined,
            .used = [_]bool{false} ** 16,
        };
    }

    pub inline fn allocate(self: *TensorPool, shape: Shape, dtype: DataType) ?*Tensor {
        for (0..16) |i| {
            if (!self.used[i]) {
                self.used[i] = true;
                self.tensors[i] = Tensor.init(shape, dtype);
                return &self.tensors[i];
            }
        }
        return null; // Pool full
    }

    pub inline fn deallocate(self: *TensorPool, tensor: *Tensor) void {
        for (0..16) |i| {
            if (&self.tensors[i] == tensor) {
                self.used[i] = false;
                break;
            }
        }
    }

    pub inline fn getAvailableCount(self: TensorPool) usize {
        var count: usize = 0;
        for (self.used) |used| {
            if (!used) count += 1;
        }
        return count;
    }
};

// Test functions
test "tensor basics" {
    const shape = Shape.init(&.{ 2, 3 });
    const tensor = Tensor.init(shape, .f32);

    try std.testing.expectEqual(@as(u8, 2), tensor.shape.rank);
    try std.testing.expectEqual(@as(usize, 6), tensor.totalElements());
    try std.testing.expectEqual(@as(usize, 24), tensor.size());
}

test "tensor operations" {
    const shape = Shape.init(&.{ 2, 2 });

    // Create tensors with data
    var a = Tensor.init(shape, .f32);
    var b = Tensor.init(shape, .f32);

    const a_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b_data = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    try a.setData(std.mem.sliceAsBytes(&a_data));
    try b.setData(std.mem.sliceAsBytes(&b_data));

    // Test addition
    const sum = try a.add(b);
    const sum_data = std.mem.bytesAsSlice(f32, sum.getData());
    try std.testing.expectEqual(@as(f32, 6.0), sum_data[0]);
    try std.testing.expectEqual(@as(f32, 8.0), sum_data[1]);
    try std.testing.expectEqual(@as(f32, 10.0), sum_data[2]);
    try std.testing.expectEqual(@as(f32, 12.0), sum_data[3]);
}

test "tensor pool" {
    var pool = TensorPool.init();
    const shape = Shape.init(&.{ 2, 2 });

    // Allocate tensors
    const tensor1 = pool.allocate(shape, .f32);
    const tensor2 = pool.allocate(shape, .f32);

    try std.testing.expect(tensor1 != null);
    try std.testing.expect(tensor2 != null);
    try std.testing.expectEqual(@as(usize, 14), pool.getAvailableCount());

    // Deallocate
    if (tensor1) |t| pool.deallocate(t);
    try std.testing.expectEqual(@as(usize, 15), pool.getAvailableCount());
}
