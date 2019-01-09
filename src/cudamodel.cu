#include "cudamodel.h"
#include "utils.h"

#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace fasttext {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

std::mutex CudaModel::initmtx_;
bool CudaModel::inited_ = false;
thrust::device_vector<real>* CudaModel::d_wi_;
thrust::device_vector<real>* CudaModel::d_wo_;
thrust::device_vector<real>* CudaModel::d_t_sigmoid_;
thrust::device_vector<real>* CudaModel::d_t_log_;
thrust::device_vector<real>* CudaModel::d_negatives_;
thrust::device_vector<int32_t>* CudaModel::d_oneVsAll_target_;
thrust::device_vector<real>* CudaModel::d_total_loss_;
thrust::device_vector<unsigned long long int>* CudaModel::d_nexamples_;

CudaModel::CudaModel(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Args> args,
    int32_t seed)
    : Model(wi, wo, args, seed),
    d_hidden_(args->dim, 0.0),
    d_output_(wo->size(0), 0.0),
    d_softmax_output_(wo->size(0), 0.0),
    d_grad_(args->dim, 0.0),
    inputpos_(0),
    targetpos_(0) {
  std::lock_guard<std::mutex> lck(initmtx_);
  if( !inited_ ) {
    d_wi_ = new thrust::device_vector<real>(wi_->vector());
    d_wo_ = new thrust::device_vector<real>(wo_->vector());
    d_t_sigmoid_ = new thrust::device_vector<real>(t_sigmoid_);
    d_t_log_ = new thrust::device_vector<real>(t_log_);
    d_negatives_ = new thrust::device_vector<real>(negatives_);
    d_total_loss_ = new thrust::device_vector<real>(1, 0.0);
    d_nexamples_ = new thrust::device_vector<unsigned long long int>(1, 1);
    if(args->loss == loss_name::ova) {
      thrust::host_vector<real> h_oneVsAll_target(osz_);
      for(int32_t i=0; i<osz_; i++ )
        h_oneVsAll_target[i] = i;
      d_oneVsAll_target_ = new thrust::device_vector<int32_t>(h_oneVsAll_target);
    }
    assert(thrust::raw_pointer_cast(d_wi_->data()));
    assert(thrust::raw_pointer_cast(d_wo_->data()));
    assert(thrust::raw_pointer_cast(d_t_sigmoid_->data()));
    assert(thrust::raw_pointer_cast(d_t_log_->data()));
    assert(d_negatives_->size()==negatives_.size());
    assert(args->dim<1024); // Max thread per block of CUDA
    inited_ = true;
  }
  stream_ = cudaStreamPerThread;
  cudnnCreate(&cudnn_);
  cudnnCreateTensorDescriptor(&cudnn_output_desc_);
  cudnnCreateTensorDescriptor(&cudnn_hidden_desc_);
  cudnnCreateTensorDescriptor(&cudnn_wi_desc_);
  cudnnSetTensor4dDescriptor(cudnn_output_desc_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 1, 1, d_output_.size());
  cudnnSetTensor4dDescriptor(cudnn_hidden_desc_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 1, 1, args->dim);
  cudnnSetTensor4dDescriptor(cudnn_wi_desc_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 1, wi->size(0), wi->size(1));
  cudnnSetStream(cudnn_, stream_);
}

CudaModel::~CudaModel() {
  flush();
  cudaStreamSynchronize(stream_);
  cudnnDestroyTensorDescriptor(cudnn_output_desc_);
  cudnnDestroyTensorDescriptor(cudnn_hidden_desc_);
  cudnnDestroyTensorDescriptor(cudnn_wi_desc_);
  cudnnDestroy(cudnn_);
}

real CudaModel::getLoss() {
#ifndef _COMPARE_WITH_CPU_
  nexamples_++;
#endif
  if( nexamples_%100 == 0 ) {
    thrust::host_vector<real> loss(*d_total_loss_);
    thrust::host_vector<unsigned long long int> nexamples(*d_nexamples_);
    loss_ = loss[0] / nexamples[0];
  }
  return loss_;
}

void CudaModel::CudaCleanup(std::shared_ptr<Matrix> wi, std::shared_ptr<Matrix> wo) {
  thrust::copy(d_wi_->begin(), d_wi_->end(), wi->vector().begin());
  thrust::copy(d_wo_->begin(), d_wo_->end(), wo->vector().begin());
  delete d_wi_;
  delete d_wo_;
  delete d_t_sigmoid_;
  delete d_t_log_;
  delete d_negatives_;
  delete d_oneVsAll_target_;
  delete d_total_loss_;
  delete d_nexamples_;
  cudaDeviceReset();
}

const real epsilon = 0.00001f;
void CudaModel::verify(const char* prefix) {
  assert(args_->thread==1);
  printf("\n");
  thrust::host_vector<real> h_hidden(d_hidden_);
  for(size_t i=0; i<hidden_.size(); i++)
    if( fabs(h_hidden[i]-hidden_[i])>epsilon )
      printf("%s hidden %lu not match: (h)%f (d)%f\n", prefix, i, hidden_[i], h_hidden[i]);
  thrust::host_vector<real> h_grad(d_grad_);
  for(size_t i=0; i<grad_.size(); i++)
    if( fabs(h_grad[i]-grad_[i])>epsilon )
      printf("%s grad %lu not match: (h)%f (d)%f\n", prefix, i, grad_[i], h_grad[i]);
  thrust::host_vector<real> h_output(args_->loss == loss_name::softmax?d_softmax_output_:d_output_);
  for(size_t i=0; i<output_.size(); i++)
    if( fabs(h_output[i]-output_[i])>epsilon )
      printf("%s output %lu not match: (h)%f (d)%f\n", prefix, i, output_[i], h_output[i]);
  thrust::host_vector<real> h_wi(*d_wi_);
  for(size_t i=0; i<wi_->vector().size(); i++)
    if( fabs(h_wi[i]-wi_->vector()[i])>epsilon )
      printf("%s wi %lu not match: (h)%f (d)%f\n", prefix, i, wi_->vector()[i], h_wi[i]);
  thrust::host_vector<real> h_wo(*d_wo_);
  for(size_t i=0; i<wo_->vector().size(); i++)
    if( fabs(h_wo[i]-wo_->vector()[i])>epsilon )
      printf("%s wo %lu not match: (h)%f (d)%f\n", prefix, i, wo_->vector()[i], h_wo[i]);
  thrust::host_vector<real> loss(*d_total_loss_);
  if( fabs(loss[0]-loss_)>epsilon )
    printf("loss not match: (h)%f, (d)%f\n", loss_, loss[0]);
  thrust::host_vector<unsigned long long int> nexamples(*d_nexamples_);
  if( nexamples[0]!=nexamples_ )
    printf("samples not match: (h)%lu, (d)%lu\n", nexamples_, nexamples[0]);
  printf("\n%s end\n", prefix);
}

__global__
void CudacomputeHidden(int32_t* input, size_t input_n, real* hidden, real* wi) {
  int input_idx = blockIdx.y*blockDim.x + threadIdx.x;
  int hidden_idx = blockIdx.x;

  __shared__ real sum;
  if( threadIdx.x==0 )
    sum = 0.0;
  __syncthreads();

  if( input_idx < input_n ) {
    atomicAdd(&sum, wi[input[input_idx]*gridDim.x+hidden_idx]);
    __syncthreads();

    if( threadIdx.x==0 )
      atomicAdd(&hidden[hidden_idx], sum);
  }
}
__global__
void CudaAverageHidden(size_t input_n, real* hidden) {
  hidden[threadIdx.x] /= input_n;
}

__device__
real CudaLog(const real* t_log, real x) {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log[i];
}

__device__
real CudaSigmoid(const real* t_sigmoid, real x) {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

__global__
void CudabinaryLogistic(int32_t* target, Bool* label, real lr, 
  real* hidden, real* wo, real* grad, real* totalloss,
  const real* t_log, const real* t_sigmoid) {
  int target_idx = blockIdx.x;
  int hidden_idx = threadIdx.x;

  __shared__ real dotRow;
  __shared__ real score;
  __shared__ real alpha;
  if( threadIdx.x==0 )
    dotRow = 0.0;
  __syncthreads();

  atomicAdd(&dotRow, wo[target[target_idx]*blockDim.x+hidden_idx]*hidden[hidden_idx]);
  __syncthreads();

  if( threadIdx.x==0 ) {
    score = CudaSigmoid(t_sigmoid, dotRow);
    alpha = lr * (real(label[target_idx]) - score);
  }
  __syncthreads();

  atomicAdd(&grad[hidden_idx], alpha * wo[target[target_idx]*blockDim.x+hidden_idx]);
  wo[target[target_idx]*blockDim.x+hidden_idx] += alpha * hidden[hidden_idx];

  if( threadIdx.x==0 ) {
    if( label[target_idx] )
      atomicAdd(totalloss, -CudaLog(t_log, score));
    else
      atomicAdd(totalloss, -CudaLog(t_log, 1.0-score));
  }
}

void CudaModel::negativeSampling(int32_t target, real lr) {
  thrust::host_vector<Bool> h_target(args_->neg+1);
  thrust::host_vector<int32_t> h_label(args_->neg+1);
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      h_target[n] = target;
      h_label[n] = 1;
    } else {
      h_target[n] = getNegative(target);
      h_label[n] = 0;
    }
  }
  d_target_ = h_target;
  d_label_ = h_label;

  cudaMemset(thrust::raw_pointer_cast(d_grad_.data()), 0, d_grad_.size()*sizeof(real));
  CudabinaryLogistic<<<args_->neg+1, args_->dim, 0, stream_>>>(
    thrust::raw_pointer_cast(d_target_.data()),
    thrust::raw_pointer_cast(d_label_.data()),
    lr,
    thrust::raw_pointer_cast(d_hidden_.data()),
    thrust::raw_pointer_cast(d_wo_->data()),
    thrust::raw_pointer_cast(d_grad_.data()),
    thrust::raw_pointer_cast(d_total_loss_->data()),
    thrust::raw_pointer_cast(d_t_log_->data()),
    thrust::raw_pointer_cast(d_t_sigmoid_->data()));
}

void CudaModel::hierarchicalSoftmax(int32_t target, real lr) {
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  thrust::host_vector<int32_t> h_label(pathToRoot.size());
  std::vector<bool>::const_iterator it(binaryCode.begin()), itend(binaryCode.end());
  for( size_t i=0; it!=itend; it++, i++ )
    h_label[i] = (*it)?1:0;
  d_target_ = pathToRoot;
  d_label_ = h_label;

  cudaMemset(thrust::raw_pointer_cast(d_grad_.data()), 0, d_grad_.size()*sizeof(real));
  CudabinaryLogistic<<<pathToRoot.size(), args_->dim, 0, stream_>>>(
    thrust::raw_pointer_cast(d_target_.data()),
    thrust::raw_pointer_cast(d_label_.data()),
    lr, 
    thrust::raw_pointer_cast(d_hidden_.data()),
    thrust::raw_pointer_cast(d_wo_->data()),
    thrust::raw_pointer_cast(d_grad_.data()),
    thrust::raw_pointer_cast(d_total_loss_->data()),
    thrust::raw_pointer_cast(d_t_log_->data()),
    thrust::raw_pointer_cast(d_t_sigmoid_->data()));
}

__global__
void CudacomputeOutput(real* hidden, real* output, size_t output_n, real* wo) {
  int hidden_idx = blockIdx.x;
  int output_idx = blockIdx.y*blockDim.x + threadIdx.x;
  if( output_idx < output_n )
    atomicAdd(&output[output_idx], wo[output_idx*gridDim.x+hidden_idx]*hidden[hidden_idx]);
}

__global__
void CudaupdateOutput(real* grad, real* wo, real* hidden, real* softmax_output, size_t output_n, real* totalloss, const real* t_log, int32_t target, real lr) {
  int hidden_idx = blockIdx.x;
  int output_idx = blockIdx.y*blockDim.x + threadIdx.x;

  __shared__ real sum;
  if( threadIdx.x==0 )
    sum = 0.0;
  __syncthreads();

  if( blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 ) {
    atomicAdd(totalloss, -CudaLog(t_log, softmax_output[target]));
  }

  if( output_idx < output_n ) {
    real label = (output_idx==target)?1.0:0.0;
    real alpha = lr * (label - softmax_output[output_idx]);
    atomicAdd(&sum, alpha*wo[output_idx*gridDim.x+hidden_idx]);
    atomicAdd(&wo[output_idx*gridDim.x+hidden_idx], alpha*hidden[hidden_idx]);
  }
  __syncthreads();

  if( threadIdx.x==0 )
    atomicAdd(&grad[hidden_idx], sum);
}

void CudaModel::softmax(int32_t target, real lr) {
  dim3 DimBlock(256, 1, 1);
  dim3 DimGrid(args_->dim, (output_.size()+DimBlock.x-1)/DimBlock.x, 1);
  cudaMemset(thrust::raw_pointer_cast(d_output_.data()), 0, d_output_.size()*sizeof(real));
  CudacomputeOutput<<<DimGrid, DimBlock, 0, stream_>>>(
    thrust::raw_pointer_cast(d_hidden_.data()),
    thrust::raw_pointer_cast(d_output_.data()),
    d_output_.size(),
    thrust::raw_pointer_cast(d_wo_->data()));
   
  static const float one = 1.0;
  static const float zero = 0.0;
  cudnnSoftmaxForward(cudnn_, cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE, 
    &one, cudnn_output_desc_, thrust::raw_pointer_cast(d_output_.data()), 
    &zero, cudnn_output_desc_, thrust::raw_pointer_cast(d_softmax_output_.data()));

  CudaupdateOutput<<<DimGrid, DimBlock, 0, stream_>>>(
    thrust::raw_pointer_cast(d_grad_.data()),
    thrust::raw_pointer_cast(d_wo_->data()),
    thrust::raw_pointer_cast(d_hidden_.data()),
    thrust::raw_pointer_cast(d_softmax_output_.data()),
    d_softmax_output_.size(),
    thrust::raw_pointer_cast(d_total_loss_->data()),
    thrust::raw_pointer_cast(d_t_log_->data()),
    target, lr);
}

__global__
void CudaUpdateOnVsAllLabel(const int32_t* target, const size_t target_n, Bool* label, size_t label_n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if( idx >= target_n )
    return;
  if( target[idx]<label_n )
    label[target[idx]] = 1;
}

void CudaModel::oneVsAll(int32_t* d_target, int32_t d_target_n, real lr) {
  int threadCnt = std::min<size_t>(1024, d_target_n);
  int blockCnt = (d_target_n+threadCnt-1)/threadCnt;
  thrust::host_vector<Bool> h_label(osz_, 0);
  d_label_ = h_label;
  CudaUpdateOnVsAllLabel<<<blockCnt, threadCnt, 0, stream_>>>(
    d_target, d_target_n,
    thrust::raw_pointer_cast(d_label_.data()),
    osz_);

  cudaMemset(thrust::raw_pointer_cast(d_grad_.data()), 0, d_grad_.size()*sizeof(real));
  CudabinaryLogistic<<<d_oneVsAll_target_->size(), args_->dim, 0, stream_>>>(
    thrust::raw_pointer_cast(d_oneVsAll_target_->data()),
    thrust::raw_pointer_cast(d_label_.data()),
    lr,
    thrust::raw_pointer_cast(d_hidden_.data()),
    thrust::raw_pointer_cast(d_wo_->data()),
    thrust::raw_pointer_cast(d_grad_.data()),
    thrust::raw_pointer_cast(d_total_loss_->data()),
    thrust::raw_pointer_cast(d_t_log_->data()),
    thrust::raw_pointer_cast(d_t_sigmoid_->data()));
}

void CudaModel::computeLoss(
    int32_t target,
    int32_t* d_target, int32_t d_target_n,
    real lr) {
  if (args_->loss == loss_name::ns) {
    negativeSampling(target, lr);
  } else if (args_->loss == loss_name::hs) {
    hierarchicalSoftmax(target, lr);
  } else if (args_->loss == loss_name::softmax) {
    softmax(target, lr);
  } else if (args_->loss == loss_name::ova) {
    oneVsAll(d_target, d_target_n, lr);
  } else {
    throw std::invalid_argument("Unhandled loss function for this model.");
  }
}

__global__
void CudacomputeInput(int32_t* input, size_t input_n, real* grad, real* wi, unsigned long long int* nexamples, bool is_sup) {
  int input_idx = blockIdx.x;
  int grad_idx = threadIdx.x;

  if( blockIdx.x==0 && threadIdx.x==0 ) {
    atomicAdd(nexamples, 1);
  }
  if( is_sup )
    grad[grad_idx] /= (real)(input_n);

  wi[input[input_idx]*blockDim.x+grad_idx] += grad[grad_idx];
}

void CudaModel::computeInput(int32_t* d_input, int32_t d_input_n) {
  CudacomputeInput<<<d_input_n, args_->dim, 0, stream_>>>(
    d_input, d_input_n,
    thrust::raw_pointer_cast(d_grad_.data()),
    thrust::raw_pointer_cast(d_wi_->data()),
    thrust::raw_pointer_cast(d_nexamples_->data()),
    args_->model==model_name::sup);
}

void CudaModel::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr) {
  if (input.size() == 0 || targets.size()==0 ) {
    return;
  }

  inputbuf_.insert(inputbuf_.end(), input.begin(), input.end());
  inputpos_ += input.size();
  inputbufpos_.push_back(inputpos_);
  if(args_->loss == loss_name::ova) {
    targetbuf_.insert(targetbuf_.end(), targets.begin(), targets.end());
    targetpos_ += targets.size();
    targetbufpos_.push_back(targetpos_);
  } else {
    target_.push_back(targets[targetIndex]);
  }
  lrbuf_.push_back(lr);

  if( inputbufpos_.size() >= args_->batchsize )
    flush();
#ifdef _COMPARE_WITH_CPU_
  // compute by gpu
  flush();

  // compute by cpu
  Model::update(input, targets, targetIndex, lr);
  
  // compare
  verify("after computeInput");
#endif
}

void CudaModel::flush() {
  size_t cur_input = 0;
  size_t cur_target = 0;
  thrust::device_vector<int32_t> d_inputbuf(inputbuf_.size());
  thrust::device_vector<int32_t> d_targetbuf(targetbuf_.size());
  int32_t* p_input = thrust::raw_pointer_cast(d_inputbuf.data());
  int32_t* p_target = thrust::raw_pointer_cast(d_targetbuf.data());
  cudaMemcpyAsync(p_input, inputbuf_.data(), inputbuf_.size()*sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
  cudaMemcpyAsync(p_target, targetbuf_.data(), targetbuf_.size()*sizeof(int32_t), cudaMemcpyHostToDevice, stream_);
  thrust::host_vector<int32_t>::const_iterator it_inputpos(inputbufpos_.begin()), it_inputposend(inputbufpos_.end());
  thrust::host_vector<int32_t>::const_iterator it_targetpos(targetbufpos_.begin());
  thrust::host_vector<int32_t>::const_iterator it_target(target_.begin());
  thrust::host_vector<real>::const_iterator it_lr(lrbuf_.begin());
  while( it_inputpos != it_inputposend ) {
    int32_t* d_input = p_input + cur_input;
    int32_t* d_target = p_target + cur_target;
    int32_t d_input_n = *it_inputpos - cur_input;
    real lr = *it_lr;
    int32_t d_target_n = 0;
    int32_t target = 0;
    if( args_->loss==loss_name::ova )
      d_target_n = *it_targetpos - cur_target;
    else
      target = *it_target;

    update_internal(d_input, d_input_n, d_target, d_target_n, target, lr);

    cur_input += d_input_n;
    cur_target += d_target_n;
    it_inputpos++;
    if( args_->loss==loss_name::ova )
      it_targetpos++;
    else
      it_target++;
    it_lr++;
  }

  inputpos_ = 0;
  targetpos_ = 0;
  inputbuf_.clear();
  targetbuf_.clear();
  inputbufpos_.clear();
  targetbufpos_.clear();
  target_.clear();
  lrbuf_.clear();
}

void CudaModel::computeHidden(int32_t* d_input, int32_t d_input_n) {
  cudaMemset(thrust::raw_pointer_cast(d_hidden_.data()), 0, d_hidden_.size()*sizeof(real));
  dim3 DimBlock(std::min<int32_t>(256, d_input_n), 1, 1);
  dim3 DimGrid(args_->dim, (d_input_n+DimBlock.x-1)/DimBlock.x, 1);
  CudacomputeHidden<<<DimGrid, DimBlock, 0, stream_>>>(
    d_input, d_input_n,
    thrust::raw_pointer_cast(d_hidden_.data()),
    thrust::raw_pointer_cast(d_wi_->data()));

  CudaAverageHidden<<<1, args_->dim, 0, stream_>>>(d_input_n, thrust::raw_pointer_cast(d_hidden_.data()));
}

void CudaModel::update_internal(
    int32_t* d_input, int32_t d_input_n,
    int32_t* d_target, int32_t d_target_n,
    int32_t target,
    real lr) {
  computeHidden(d_input, d_input_n);
  computeLoss(target, d_target, d_target_n, lr);
  computeInput(d_input, d_input_n);
}

} // namespace fasttext
