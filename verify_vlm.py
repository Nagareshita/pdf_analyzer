import sys
import torch
import transformers
import accelerate
from pathlib import Path

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

try:
    from vlm.model_manager import ModelManager
    print("\nInitializing ModelManager...")
    mm = ModelManager()
    print("Running setup_model()...")
    success = mm.setup_model(progress_callback=lambda x: print(f"Progress: {x}"))
    if success:
        print("✅ Model setup successful!")
    else:
        print("❌ Model setup failed!")
except Exception as e:
    print(f"❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()
