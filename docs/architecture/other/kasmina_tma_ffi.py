# kasmina_tma_ffi.py
import ctypes
import enum
import platform
import torch
import warnings

# =============================================================================
#
#  CUDA Driver API Definitions for TMA via ctypes
#  (Based on CUDA 12.x Toolkit Documentation)
#
# =============================================================================

class CUDAResult(enum.IntEnum):
    """Subset of CUDA driver API result codes."""
    CUDA_SUCCESS = 0
    CUDA_ERROR_NOT_FOUND = 500
    CUDA_ERROR_INVALID_VALUE = 1

# --- Enums for cuTensorMapEncodeTiled ---

class CUTensorMapDataType(enum.IntEnum):
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0
    CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1
    CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2
    CU_TENSOR_MAP_DATA_TYPE_UINT64 = 3
    CU_TENSOR_MAP_DATA_TYPE_INT32 = 4
    CU_TENSOR_MAP_DATA_TYPE_INT64 = 5
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9

class CUTensorMapSwizzle(enum.IntEnum):
    CU_TENSOR_MAP_SWIZZLE_NONE = 0
    CU_TENSOR_MAP_SWIZZLE_32B = 1
    CU_TENSOR_MAP_SWIZZLE_64B = 2
    CU_TENSOR_MAP_SWIZZLE_128B = 3

class CUTensorMapInterleave(enum.IntEnum):
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0

class CUTensorMapL2Promotion(enum.IntEnum):
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B = 1
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3

class CUTensorMapFloatOobFill(enum.IntEnum):
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0
    CU_TENSOR_MAP_FLOAT_OOB_FILL_ZEROS = 1
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN = 2

# --- Structures ---

_CU_TENSOR_MAP_MAX_RANK = 16

class CUTensorMap(ctypes.Structure):
    """
    The opaque TMA descriptor object.
    The driver populates this structure. Its size is fixed (128 bytes in CUDA 12).
    """
    _fields_ = [
        ("internal", ctypes.c_uint8 * 128)
    ]

# --- CUDA Driver Loader ---

class CUDADriver:
    """
    Singleton class to load the CUDA driver library and access its functions.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CUDADriver, cls).__new__(cls)
            cls._instance._init_driver()
        return cls._instance

    def _init_driver(self):
        self.lib = None
        self.cuTensorMapEncodeTiled = None

        try:
            if platform.system() == "Windows":
                self.lib = ctypes.WinDLL("nvcuda.dll")
            else:
                self.lib = ctypes.CDLL("libcuda.so", mode=ctypes.RTLD_GLOBAL)
        except (OSError, TypeError) as e:
            warnings.warn(f"Could not load CUDA driver library. TMA will be disabled. Error: {e}")
            return

        # --- Define cuTensorMapEncodeTiled function signature ---
        try:
            self.cuTensorMapEncodeTiled = self.lib.cuTensorMapEncodeTiled
            self.cuTensorMapEncodeTiled.restype = CUDAResult
            self.cuTensorMapEncodeTiled.argtypes = [
                ctypes.POINTER(CUTensorMap),      # tensorMap
                CUTensorMapDataType,              # dataType
                ctypes.c_uint32,                  # rank
                ctypes.c_void_p,                  # globalAddress
                ctypes.POINTER(ctypes.c_uint64),  # globalDim
                ctypes.POINTER(ctypes.c_uint64),  # globalStrides
                ctypes.POINTER(ctypes.c_uint32),  # boxDim
                CUTensorMapSwizzle,               # elementWiseSwizzle
                CUTensorMapInterleave,            # interleave
                CUTensorMapL2Promotion,           # l2Promotion
                CUTensorMapFloatOobFill           # oobFill
            ]
        except AttributeError as e:
            warnings.warn(f"CUDA driver function 'cuTensorMapEncodeTiled' not found. "
                          f"Your driver may be too old. TMA will be disabled. Error: {e}")
            self.cuTensorMapEncodeTiled = None

    @property
    def is_available(self):
        return self.cuTensorMapEncodeTiled is not None

# Helper function to map PyTorch dtypes to CUDA TMA dtypes
DTYPE_MAP = {
    torch.uint8: CUTensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int32: CUTensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64: CUTensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.float16: CUTensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16: CUTensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float32: CUTensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float64: CUTensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
}