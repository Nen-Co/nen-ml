const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library
    const lib = b.addStaticLibrary(.{
        .name = "nen-ml",
        .root_source_file = .{ .cwd_relative = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(lib);

    // Main executable
    const exe = b.addExecutable(.{
        .name = "nen-ml",
        .root_source_file = .{ .cwd_relative = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    exe.addModule("nen_ml", lib);

    b.installArtifact(exe);

    // Tests
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .cwd_relative = "src/lib.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Examples
    const tensor_example = b.addExecutable(.{
        .name = "tensor-example",
        .root_source_file = .{ .cwd_relative = "examples/tensor_demo.zig" },
        .target = target,
        .optimize = optimize,
    });

    tensor_example.addModule("nen_ml", lib);

    const run_tensor_example = b.addRunArtifact(tensor_example);

    const tensor_step = b.step("run-tensor-example", "Run tensor demo");
    tensor_step.dependOn(&run_tensor_example.step);

    const nn_example = b.addExecutable(.{
        .name = "nn-example",
        .root_source_file = .{ .cwd_relative = "examples/nn_demo.zig" },
        .target = target,
        .optimize = optimize,
    });

    nn_example.addModule("nen_ml", lib);

    const run_nn_example = b.addRunArtifact(nn_example);

    const nn_step = b.step("run-nn-example", "Run neural network demo");
    nn_step.dependOn(&run_nn_example.step);

    const workflow_example = b.addExecutable(.{
        .name = "workflow-example",
        .root_source_file = .{ .cwd_relative = "examples/workflow_demo.zig" },
        .target = target,
        .optimize = optimize,
    });

    workflow_example.addModule("nen_ml", lib);

    const run_workflow_example = b.addRunArtifact(workflow_example);

    const workflow_step = b.step("run-workflow-example", "Run workflow demo");
    workflow_step.dependOn(&run_workflow_example.step);

    // All examples
    const examples_step = b.step("run-examples", "Run all examples");
    examples_step.dependOn(&run_tensor_example.step);
    examples_step.dependOn(&run_nn_example.step);
    examples_step.dependOn(&run_workflow_example.step);
}
