// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#include <fstream>
#include "test_fixture.h"
#include "file_util.h"
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(CApiTest, model_from_array) {
  const char* model_path = "testdata/matmul_1.onnx";
  std::vector<char> buffer;
  {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file)
      throw std::runtime_error("Error reading model");
    buffer.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), buffer.size()))
      throw std::runtime_error("Error reading model");
  }

  Ort::SessionOptions so;
  Ort::Session session(*ort_env.get(), buffer.data(), buffer.size(), so);

#ifdef USE_CUDA
  // test with CUDA provider when using onnxruntime as dll
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0));
  Ort::Session session_cuda(*ort_env.get(), buffer.data(), buffer.size(), so);
#endif
}

TEST(CApiTest, TestModelWithExternalDataFromArray) {
  const char* model_path = "testdata/layoutlm-doc_seq_rel-large.graph.onnx";
  const char* external_data_path = "testdata/layoutlm-doc_seq_rel-large.weight";
  const char* external_data_name = "layoutlm-doc_seq_rel-large.weight";

  auto read_buffer = [](const char* file_path) {
    std::string buffer;
    std::ifstream file(file_path, std::ios::binary);
    if (!file)
      ORT_THROW("Error reading model");
    std::ostringstream oss;
    oss << file.rdbuf();
    buffer = oss.str();
    file.close();
    return buffer;
  };

  std::string model_buffer = read_buffer(model_path), external_data_buffer = read_buffer(external_data_path);

  std::vector<std::string> external_data_names = {external_data_name};
  std::vector<const void*> external_data_buffers = {external_data_buffer.data()};

  auto create_session = [&](Ort::SessionOptions& so) {
    Ort::Session session(*ort_env.get(),
                        model_buffer.data(),
                        model_buffer.size(),
                        external_data_names.data(),
                        external_data_buffers.data(),
                        external_data_buffers.size(),
                        so);
  };

  Ort::SessionOptions so;
  create_session(so);
#ifdef USE_CUDA
  // test with CUDA provider when using onnxruntime as dll
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(so, 0));
  create_session(so);
#endif
}
}  // namespace test
}  // namespace onnxruntime
