/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"

#include <memory>
#include <vector>

#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/device.h"  // IWYU pragma: keep
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

extern "C" {
// from test_pluggable_device.cc
void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status);
}

class PluggableDeviceFactoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    PluggableDeviceInit_Api api;
    api.init_plugin_fn = [](SE_PlatformRegistrationParams* const params,
                            TF_Status* const status) {
      SE_InitPlugin(params, status);
      params->platform->name = "MY_PLATFORM";
      params->platform->use_bfc_allocator = true;
    };
    TF_ASSERT_OK(RegisterPluggableDevicePlugin(&api));

    // Initialize executors for the test platform.
    auto platform_or =
        stream_executor::PlatformManager::PlatformWithName("MY_PLATFORM");
    TF_ASSERT_OK(platform_or);
    stream_executor::Platform* platform = platform_or.value();
    for (int i = 0; i < platform->VisibleDeviceCount(); ++i) {
      TF_ASSERT_OK(platform->ExecutorForDevice(i));
    }
  }

  SessionOptions MakeSessionOptions(
      const std::vector<std::vector<float>>& memory_limit_mb) {
    SessionOptions options;
    ConfigProto* config = &options.config;
    GPUOptions* gpu_options = config->mutable_pluggable_device_options();
    if (!memory_limit_mb.empty()) {
      for (int i = 0; i < memory_limit_mb.size(); ++i) {
        auto virtual_devices =
            gpu_options->mutable_experimental()->add_virtual_devices();
        for (float mb : memory_limit_mb[i]) {
          virtual_devices->add_memory_limit_mb(mb);
        }
      }
    }
    return options;
  }
};

TEST_F(PluggableDeviceFactoryTest, VirtualDevicesMemoryLimitTest) {
  SessionOptions opts = MakeSessionOptions({{123, 456}, {789}});
  std::vector<std::unique_ptr<Device>> devices;
  PluggableDeviceFactory factory("MY_DEVICE", "MY_PLATFORM");
  TF_ASSERT_OK(
      factory.CreateDevices(opts, "/job:localhost/replica:0/task:0", &devices));
  EXPECT_EQ(devices.size(), 3);
  EXPECT_EQ(devices[0]->attributes().memory_limit(), 123 << 20);
  EXPECT_EQ(devices[1]->attributes().memory_limit(), 456 << 20);
  EXPECT_EQ(devices[2]->attributes().memory_limit(), 789 << 20);
}

}  // namespace
}  // namespace tensorflow
