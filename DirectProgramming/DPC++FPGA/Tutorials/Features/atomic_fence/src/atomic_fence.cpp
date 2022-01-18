//==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <chrono>
#include <numeric>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;

constexpr size_t vector_size = 10000; // Size of the input vector
constexpr double kNs = 1e9;           // number of nanoseconds in a second
constexpr bool READY = true;

// Forward declare the kernel names in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <bool use_fences> class ProducerKernel;

template <bool use_fences> class ConsumerKernel;

template <bool use_fences> class some_pipe;

template <bool use_fences>
void launchKernels(queue &q, const std::vector<int> &in,
                   std::vector<int> &out) {
  using my_pipe = ext::intel::pipe<some_pipe<use_fences>, bool>;

  assert(in.size() == vector_size);
  assert(out.size() == vector_size);

  // Allocate the device memory
  int *in_ptr = malloc_device<int>(vector_size, q);
  int *out_ptr = malloc_device<int>(vector_size, q);

  // A buffer shared by consumer and producer to exchange data
  int *buffer_ptr = malloc_device<int>(vector_size, q);

  // Ensure we successfully allocated the device memory
  if (in_ptr == nullptr)
    std::cerr << "ERROR: failed to allocate space for 'in_ptr'\n";

  if (out_ptr == nullptr)
    std::cerr << "ERROR: failed to allocate space for 'out_ptr'\n";

  if (buffer_ptr == nullptr)
    std::cerr << "ERROR: failed to allocate space for 'buffer_ptr'\n";

  // Copy host input data to the device's memory
  auto copy_host_to_device_event =
      q.memcpy(in_ptr, in.data(), vector_size * sizeof(int));

  // Launch the the consumer kernel
  auto consumer_event = q.submit([&](handler &h) {
    h.single_task<ConsumerKernel<use_fences>>([=
    ]() [[intel::kernel_args_restrict]] {
      // Create device pointers to explicitly inform the compiler these
      // pointer reside in the device's address space
      device_ptr<int> buffer_ptr_d(buffer_ptr);
      device_ptr<int> out_ptr_d(out_ptr);

      // This kernel must wait on the blocking pipe_read until notified by
      // producer
      int ready = my_pipe::read();

      // Use atomic_fence to ensure memory ordering
      if constexpr (use_fences)
        atomic_fence(memory_order::seq_cst, memory_scope::device);

      for (size_t i = vector_size - 1; i >= 0; i--)
        out_ptr_d[i] = buffer_ptr_d[i];
    });
  });

  // Launch the the producer kernel
  auto producer_event = q.submit([&](handler &h) {
    // This kernel must wait until the data is copied from the host's to
    // the device's memory
    h.depends_on(copy_host_to_device_event);

    h.single_task<ProducerKernel<use_fences>>([=
    ]() [[intel::kernel_args_restrict]] {
      // Create device pointers to explicitly inform the compiler these
      // Pointer reside in the device's address space
      device_ptr<int> in_ptr_d(in_ptr);
      device_ptr<int> buffer_ptr_d(buffer_ptr);

      for (size_t i = 0; i < vector_size; i++)
        buffer_ptr_d[i] = in_ptr_d[i] * i;

      // Use atomic_fence to ensure memory ordering
      if constexpr (use_fences)
        atomic_fence(memory_order::seq_cst, memory_scope::device);

      // Notify the consumer to start data processing
      my_pipe::write(READY);
    });
  });

  // Copy output data back from device to host
  auto copy_device_to_host_event = q.submit([&](handler &h) {
    // We cannot copy the output data from the device's to the host's memory
    // until the consumer kernel has finished
    h.depends_on(consumer_event);
    h.memcpy(out.data(), out_ptr, vector_size * sizeof(int));
  });

  // Wait for copy back to finish
  copy_device_to_host_event.wait();

  // Free the device memory
  // Note that these are calls to sycl::free()
  free(in_ptr, q);
  free(buffer_ptr, q);
  free(out_ptr, q);
}

int main() {
  // Host and kernel profiling
  event e;
  ulong t1_kernel, t2_kernel;
  double time_kernel;

  // Create input and output vectors
  std::vector<int> in, out_fpga_with_fence, out_fpga_without_fence, out_cpu;
  in.resize(vector_size);
  out_fpga_with_fence.resize(vector_size);
  out_fpga_without_fence.resize(vector_size);
  out_cpu.resize(vector_size);

  // Generate some random input data
  std::generate(in.begin(), in.end(), [=] { return int(rand() % 100); });

// Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  ext::intel::fpga_emulator_selector device_selector;
#else
  ext::intel::fpga_selector device_selector;
#endif
  try {
    auto prop_list =
        sycl::property_list{sycl::property::queue::enable_profiling()};

    sycl::queue q(device_selector, dpc_common::exception_handler, prop_list);

    std::cout << "\nVector size: " << vector_size << "\n";

    launchKernels<true>(q, in, out_fpga_with_fence);
    launchKernels<false>(q, in, out_fpga_without_fence);
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  // Compute the reference solution
  for (size_t i = 0; i < vector_size; ++i)
    out_cpu[i] = in[i] * i;

  // Verify output and print pass/fail
  bool match_with_fences = true;
  bool mismatch_without_fences = false;
  int num_errors = 0;

  for (int b = 0; b < 100; b++) {
    std::cout << " out_cpu[b]: " << out_cpu[b] << "\n";
    std::cout << " out_fpga_with_fence[b]: " << out_fpga_with_fence[b] << "\n";
    std::cout << " out_fpga_without_fence[b]: " << out_fpga_without_fence[b] << "\n";
  }

  for (int b = 0; b < vector_size; b++) {
    if (num_errors < 10 && out_fpga_with_fence[b] != out_cpu[b]) {
      match_with_fences = false;
      std::cerr << " (mismatch, expected " << out_cpu[b] << ")\n";
      num_errors++;
    }
  }

  for (int b = 0; b < vector_size; b++) {
    if (out_fpga_without_fence[b] != out_cpu[b]) {
      mismatch_without_fences = true;
      break;
    }
  }

  if (match_with_fences) {
    std::cout
        << "Results are correct when fences are used, as expected. PASSED.\n\n";
  } else {
    std::cerr << "Results are incorrect when fences are used. This is "
                 "unexpected. FAILED.\n\n";
    return 1;
  }

#if !defined(FPGA_EMULATOR)
  if (mismatch_without_fences) {
    std::cout << "Results are incorrect when fences are not used, as expected. "
                 "PASSED.\n\n";
  } else {
    std::cerr << "Results are correct when fences are not used. This is "
                 "unexpected. FAILED.\n\n";
    return 1;
  }
#endif

  return 0;
}
