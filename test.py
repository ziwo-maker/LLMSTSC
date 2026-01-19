def check_torch_available():
    try:
        import torch
        print(f"torch import: OK  (version={torch.__version__})")

        # CUDA 是否可用
        cuda_ok = torch.cuda.is_available()
        print(f"cuda available: {cuda_ok}")

        if cuda_ok:
            idx = torch.cuda.current_device()
            print(f"cuda device: {idx}  ({torch.cuda.get_device_name(idx)})")
            print(f"cuda capability: {torch.cuda.get_device_capability(idx)}")
            print(f"cuda mem (free/total): "
                  f"{torch.cuda.mem_get_info()[0] / 1024**3:.2f}GB / {torch.cuda.mem_get_info()[1] / 1024**3:.2f}GB")

        # 简单张量运算自检
        x = torch.tensor([1.0, 2.0, 3.0])
        y = x * 2
        print("tensor op: OK", y.tolist())

        return True
    except Exception as e:
        print("torch not available:", repr(e))
        return False


if __name__ == "__main__":
    ok = check_torch_available()
    print("RESULT:", "AVAILABLE" if ok else "NOT AVAILABLE")
