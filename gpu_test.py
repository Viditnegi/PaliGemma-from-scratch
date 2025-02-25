import torch
import time

def test_gpu():
    if torch.cuda.is_available():
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. Running on CPU.")
        return

    # Create a large tensor
    tensor_size = (10000, 10000)
    
    # Test on CPU
    cpu_tensor = torch.randn(tensor_size)
    start_time = time.time()
    cpu_result = cpu_tensor @ cpu_tensor
    cpu_time = time.time() - start_time
    print(f"CPU computation time: {cpu_time:.4f} seconds")

    # Move to GPU
    gpu_tensor = cpu_tensor.to("cuda")
    torch.cuda.synchronize()  # Ensure previous operations are done
    start_time = time.time()
    gpu_result = gpu_tensor @ gpu_tensor
    torch.cuda.synchronize()  # Wait for the computation to finish
    gpu_time = time.time() - start_time
    print(f"GPU computation time: {gpu_time:.4f} seconds")

    print(f"Speedup factor: {cpu_time / gpu_time:.2f}x")

if __name__ == "__main__":
    test_gpu()
