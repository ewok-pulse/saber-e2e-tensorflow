load("//tensorflow:tensorflow.bzl", "if_oss", "tf_cc_binary")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cc_binary(
    name = "tfrt_ifrt_opt",
    testonly = 1,
    srcs = ["tfrt_ifrt_opt.cc"],
    linkstatic = 1,
    deps = [
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir/tfrt/transforms/ifrt:pack_inputs_pass",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:RegisterAllDialects",  # buildcleaner: keep
        "@llvm-project//mlir:RegisterAllPasses",  # buildcleaner: keep
        "@stablehlo//:register",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = 1,
    data = [
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [
        ":test_utilities",
        ":tfrt_ifrt_opt",
    ],
    # TODO(b/445201291): The test is failing in OSS probably due to incorrect configure path.
    # Re-enable it once the issue is fixed.
    default_tags = ["no_oss"],
    driver = "//tensorflow/compiler/mlir:run_lit.sh",
    features = [
        "--path=third_party/tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata/lit",
    ] + if_oss([
        "--path=org_tensorflow/tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata/lit",
        "--path=tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata/lit",
    ]),
    test_file_exts = ["mlir"],
)

filegroup(
    name = "testdata",
    srcs = glob(
        ["*"],
    ),
)
