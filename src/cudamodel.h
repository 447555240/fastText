#pragma once

#include <mutex>
#include <cudnn.h>
#include "model.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace fasttext {

typedef unsigned char Bool;

class CudaModel : public Model {
 public:
  CudaModel(
      std::shared_ptr<Matrix>,
      std::shared_ptr<Matrix>,
      std::shared_ptr<Args>,
      int32_t); 
  virtual ~CudaModel();
  void verify(const char* prefix);

  static void CudaCleanup(std::shared_ptr<Matrix> wi, std::shared_ptr<Matrix> wo);

  virtual void update(
      const std::vector<int32_t>&,
      const std::vector<int32_t>&,
      int32_t,
      real);
  virtual real getLoss();

 protected:
  void computeLoss(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr);
  void negativeSampling(int32_t target, real lr);
  void hierarchicalSoftmax(int32_t target, real lr);
  void softmax(int32_t target, real lr); 
  void oneVsAll(const std::vector<int32_t>& targets, real lr);
  void computeInput(thrust::device_vector<int32_t>& d_input);

 protected:
  static std::mutex initmtx_;
  static bool inited_;
  static thrust::device_vector<real>* d_wi_;
  static thrust::device_vector<real>* d_wo_;
  static thrust::device_vector<real>* d_t_sigmoid_;
  static thrust::device_vector<real>* d_t_log_;
  static thrust::device_vector<real>* d_negatives_;
  static thrust::device_vector<int32_t>* d_oneVsAll_target_;
  static thrust::device_vector<real>* d_total_loss_;
  static thrust::device_vector<unsigned long long int>* d_nexamples_;
  thrust::device_vector<real> d_hidden_;
  thrust::device_vector<real> d_output_;
  thrust::device_vector<real> d_softmax_output_;
  thrust::device_vector<real> d_grad_;
  thrust::device_vector<int32_t> d_input_;
  thrust::device_vector<Bool> d_label_;
  thrust::device_vector<int32_t> d_target_;

  thrust::host_vector<int32_t> inputbuf_;
  thrust::host_vector<int32_t> inputbufpos_;
  int32_t inputpos_;
  thrust::host_vector<int32_t> targetbuf_;
  thrust::host_vector<int32_t> targetbufpos_;
  int32_t targetpos_;
  thrust::host_vector<real> lrbuf_;

  cudaStream_t stream_;
  cudnnHandle_t cudnn_;
  cudnnTensorDescriptor_t cudnn_desc_;
};

} // namespace fasttext