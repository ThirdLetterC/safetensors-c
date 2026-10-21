/**
 * @file safetensors.h
 * @brief Embeddable C/C++ FFI header for the `safetensors-c` Rust wrapper library.
 *
 * Wraps the official Hugging Face `safetensors` Rust crate into a static library
 * (`libsafetensors_c.a` / `safetensors_c.lib`) with zero-copy deserialization,
 * cross-platform OS memory-mapping (`mmap` / `CreateFileMappingW`), and serialization.
 *
 * Windows Note:
 * When linking statically under MSVC (`cl.exe` / `link.exe`), this header automatically
 * emits `#pragma comment(lib, ...)` directives for the required Windows system libraries
 * (`ws2_32.lib`, `userenv.lib`, `bcrypt.lib`, `ntdll.lib`, `advapi32.lib`) unless
 * `SAFETENSORS_NO_AUTOLINK` is defined before including this header.
 * When linking under MinGW-w64 (`gcc` / `clang`), pass:
 *   `-lsafetensors_c -lws2_32 -luserenv -lbcrypt -lntdll -ladvapi32`
 */

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER) && !defined(SAFETENSORS_SHARED) && !defined(SAFETENSORS_NO_AUTOLINK)
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "userenv.lib")
#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "ntdll.lib")
#pragma comment(lib, "advapi32.lib")
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(SAFETENSORS_SHARED)
    #if defined(SAFETENSORS_EXPORTS)
      #define SAFETENSORS_API __declspec(dllexport)
    #else
      #define SAFETENSORS_API __declspec(dllimport)
    #endif
  #else
    #define SAFETENSORS_API
  #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define SAFETENSORS_API __attribute__((visibility("default")))
#else
  #define SAFETENSORS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes returned by fallible `safetensors_*` C API functions.
 */
typedef enum SafetensorsStatus {
    /** Operation completed without error. */
    SAFETENSORS_OK = 0,
    /** A required pointer argument was NULL. */
    SAFETENSORS_ERR_NULL_POINTER = 1,
    /** A string argument was not valid UTF-8 (or UTF-16 for `*_w` functions). */
    SAFETENSORS_ERR_INVALID_UTF8 = 2,
    /** File I/O or OS memory-mapping failed. */
    SAFETENSORS_ERR_IO = 3,
    /** Header or tensor buffer validation failed during deserialization. */
    SAFETENSORS_ERR_DESERIALIZE = 4,
    /** Tensor validation or formatting failed during serialization. */
    SAFETENSORS_ERR_SERIALIZE = 5,
    /** Requested tensor name or metadata key does not exist. */
    SAFETENSORS_ERR_NOT_FOUND = 6,
    /** Requested tensor or metadata index is out of bounds. */
    SAFETENSORS_ERR_INDEX_OUT_OF_BOUNDS = 7,
    /** Unrecognized or unsupported tensor element data type (`Dtype`). */
    SAFETENSORS_ERR_INVALID_DTYPE = 8,
    /** Provided output buffer is too small. */
    SAFETENSORS_ERR_BUFFER_TOO_SMALL = 9,
    /** An unexpected internal Rust panic was caught at the FFI boundary. */
    SAFETENSORS_ERR_PANIC = 255
} SafetensorsStatus;

/**
 * @brief Element data types supported by the `safetensors` specification.
 */
typedef enum SafetensorsDtype {
    /** Boolean (8-bit, 1 byte per element) */
    SAFETENSORS_DTYPE_BOOL = 0,
    /** Unsigned 8-bit integer (`uint8_t`) */
    SAFETENSORS_DTYPE_U8 = 1,
    /** Signed 8-bit integer (`int8_t`) */
    SAFETENSORS_DTYPE_I8 = 2,
    /** 8-bit floating point (E5M2) */
    SAFETENSORS_DTYPE_F8_E5M2 = 3,
    /** 8-bit floating point (E4M3) */
    SAFETENSORS_DTYPE_F8_E4M3 = 4,
    /** 8-bit floating point (E8M0 microscaling exponent) */
    SAFETENSORS_DTYPE_F8_E8M0 = 5,
    /** 8-bit floating point (AMD FNUZ E4M3) */
    SAFETENSORS_DTYPE_F8_E4M3FNUZ = 6,
    /** 8-bit floating point (AMD FNUZ E5M2) */
    SAFETENSORS_DTYPE_F8_E5M2FNUZ = 7,
    /** Signed 16-bit integer (`int16_t`) */
    SAFETENSORS_DTYPE_I16 = 8,
    /** Unsigned 16-bit integer (`uint16_t`) */
    SAFETENSORS_DTYPE_U16 = 9,
    /** IEEE 754 half-precision 16-bit float (`f16`) */
    SAFETENSORS_DTYPE_F16 = 10,
    /** Brain floating point 16-bit (`bfloat16`) */
    SAFETENSORS_DTYPE_BF16 = 11,
    /** Signed 32-bit integer (`int32_t`) */
    SAFETENSORS_DTYPE_I32 = 12,
    /** Unsigned 32-bit integer (`uint32_t`) */
    SAFETENSORS_DTYPE_U32 = 13,
    /** IEEE 754 single-precision 32-bit float (`float`) */
    SAFETENSORS_DTYPE_F32 = 14,
    /** 64-bit complex float (two 32-bit floats: real, imag) */
    SAFETENSORS_DTYPE_C64 = 15,
    /** IEEE 754 double-precision 64-bit float (`double`) */
    SAFETENSORS_DTYPE_F64 = 16,
    /** Signed 64-bit integer (`int64_t`) */
    SAFETENSORS_DTYPE_I64 = 17,
    /** Unsigned 64-bit integer (`uint64_t`) */
    SAFETENSORS_DTYPE_U64 = 18,
    /** 4-bit microscaling float (`MXFP4`) */
    SAFETENSORS_DTYPE_F4 = 19,
    /** 6-bit microscaling float (`MXFP6 E2M3`) */
    SAFETENSORS_DTYPE_F6_E2M3 = 20,
    /** 6-bit microscaling float (`MXFP6 E3M2`) */
    SAFETENSORS_DTYPE_F6_E3M2 = 21
} SafetensorsDtype;

/**
 * @brief Opaque handle representing a loaded `.safetensors` file or buffer.
 */
typedef struct SafetensorsHandle SafetensorsHandle;

/**
 * @brief Opaque handle for incrementally building and saving a `.safetensors` file or buffer.
 */
typedef struct SafetensorsSerializer SafetensorsSerializer;

/**
 * @brief Zero-copy view of a deserialized tensor.
 *
 * All pointer fields (`name`, `shape`, `data`) are borrowed from the parent
 * `SafetensorsHandle` and remain valid until `safetensors_free(handle)` is called.
 */
typedef struct SafetensorsTensorView {
    /** Null-terminated UTF-8 tensor name. */
    const char *name;
    /** Element data type. */
    SafetensorsDtype dtype;
    /** Number of dimensions (rank). `0` for scalar tensors. */
    size_t ndim;
    /** Pointer to array of `ndim` dimension sizes (`NULL` only if `ndim == 0`). */
    const size_t *shape;
    /** Total number of elements in the tensor (product of `shape[0..ndim]`). */
    size_t num_elements;
    /** Zero-copy pointer to raw little-endian tensor bytes. */
    const void *data;
    /** Size of the tensor data buffer in bytes. */
    size_t data_len;
} SafetensorsTensorView;

/**
 * @brief Input tensor descriptor used when serializing tensors to a file or buffer.
 */
typedef struct SafetensorsTensorInput {
    /** Null-terminated UTF-8 tensor name. */
    const char *name;
    /** Element data type. */
    SafetensorsDtype dtype;
    /** Number of dimensions (rank). */
    size_t ndim;
    /** Pointer to array of `ndim` dimension sizes (may be `NULL` if `ndim == 0`). */
    const size_t *shape;
    /** Pointer to raw little-endian tensor byte buffer. */
    const void *data;
    /** Length of `data` in bytes. */
    size_t data_len;
} SafetensorsTensorInput;

/**
 * @brief Key-value metadata entry for the `__metadata__` JSON header map.
 */
typedef struct SafetensorsMetadataEntry {
    /** Null-terminated UTF-8 key string. */
    const char *key;
    /** Null-terminated UTF-8 value string. */
    const char *value;
} SafetensorsMetadataEntry;

/* =========================================================================
 * Version & Diagnostics
 * ========================================================================= */

/** Returns the static null-terminated version string of `safetensors-c` (e.g. `"0.8.0"`). */
SAFETENSORS_API const char *safetensors_version(void);

/**
 * Returns the most recent error message on the calling thread, or `NULL` if the last call
 * succeeded. Valid until the next `safetensors_*` call on the same thread.
 */
SAFETENSORS_API const char *safetensors_last_error_message(void);

/** Returns a static human-readable description string for `status`. */
SAFETENSORS_API const char *safetensors_status_str(SafetensorsStatus status);

/* =========================================================================
 * Dtype Utilities
 * ========================================================================= */

/** Returns the bit width of a single element of `dtype` (e.g., `32` for `F32`, `4` for `F4`). */
SAFETENSORS_API size_t safetensors_dtype_bitsize(SafetensorsDtype dtype);

/** Returns the byte size of a single element of `dtype`, or `0` for sub-byte types (`F4`, `F6_*`). */
SAFETENSORS_API size_t safetensors_dtype_size(SafetensorsDtype dtype);

/** Returns the canonical null-terminated string name of `dtype` (e.g., `"F32"`, `"BF16"`). */
SAFETENSORS_API const char *safetensors_dtype_name(SafetensorsDtype dtype);

/** Parses a canonical dtype string (e.g., `"F32"`, `"BF16"`, `"I64"`) into `*out_dtype`. */
SAFETENSORS_API SafetensorsStatus safetensors_dtype_from_str(
    const char *name,
    SafetensorsDtype *out_dtype
);

/* =========================================================================
 * Loading & Deserialization
 * ========================================================================= */

/**
 * Opens a `.safetensors` file using cross-platform OS memory-mapping (`mmap` on Linux/macOS,
 * `CreateFileMappingW` / `MapViewOfFile` on Windows) for zero-copy access.
 */
SAFETENSORS_API SafetensorsStatus safetensors_open_file(
    const char *path,
    SafetensorsHandle **out_handle
);

/**
 * Windows wide-character (`const uint16_t*` / `const wchar_t*`) variant of `safetensors_open_file`.
 */
SAFETENSORS_API SafetensorsStatus safetensors_open_file_w(
    const uint16_t *path_w,
    SafetensorsHandle **out_handle
);

/**
 * Opens a `.safetensors` file by reading the entire file into an owned heap buffer.
 */
SAFETENSORS_API SafetensorsStatus safetensors_open_file_buffered(
    const char *path,
    SafetensorsHandle **out_handle
);

/**
 * Deserializes a `.safetensors` byte buffer by copying `buffer[0..len]` into `out_handle`.
 * The caller may free or reuse `buffer` immediately after this call returns.
 */
SAFETENSORS_API SafetensorsStatus safetensors_deserialize_copy(
    const uint8_t *buffer,
    size_t len,
    SafetensorsHandle **out_handle
);

/**
 * Deserializes a `.safetensors` byte buffer in zero-copy mode without copying `buffer`.
 * The caller MUST keep `buffer` valid and unmodified until `safetensors_free(handle)` is called.
 */
SAFETENSORS_API SafetensorsStatus safetensors_deserialize_view(
    const uint8_t *buffer,
    size_t len,
    SafetensorsHandle **out_handle
);

/**
 * Releases all resources (including any memory map or internal buffers) held by `handle`.
 */
SAFETENSORS_API void safetensors_free(SafetensorsHandle *handle);

/* =========================================================================
 * Tensor Lookup & Inspection
 * ========================================================================= */

/** Returns the number of tensors in `handle` (or `0` if `handle` is `NULL`). */
SAFETENSORS_API size_t safetensors_num_tensors(const SafetensorsHandle *handle);

/** Returns the null-terminated tensor name at `index` (ordered by byte offset), or `NULL`. */
SAFETENSORS_API const char *safetensors_tensor_name(
    const SafetensorsHandle *handle,
    size_t index
);

/** Populates `*out_view` with a zero-copy view of the tensor at `index`. */
SAFETENSORS_API SafetensorsStatus safetensors_get_tensor_by_index(
    const SafetensorsHandle *handle,
    size_t index,
    SafetensorsTensorView *out_view
);

/** Populates `*out_view` with a zero-copy view of the tensor named `name`. */
SAFETENSORS_API SafetensorsStatus safetensors_get_tensor_by_name(
    const SafetensorsHandle *handle,
    const char *name,
    SafetensorsTensorView *out_view
);

/* =========================================================================
 * Header Metadata (__metadata__) Inspection
 * ========================================================================= */

/** Returns the number of key-value pairs in the `__metadata__` header map. */
SAFETENSORS_API size_t safetensors_num_metadata(const SafetensorsHandle *handle);

/** Returns the metadata key at `index`, or `NULL` if out of bounds. */
SAFETENSORS_API const char *safetensors_metadata_key(
    const SafetensorsHandle *handle,
    size_t index
);

/** Returns the metadata value at `index`, or `NULL` if out of bounds. */
SAFETENSORS_API const char *safetensors_metadata_value(
    const SafetensorsHandle *handle,
    size_t index
);

/** Looks up a metadata value by `key`, returning a null-terminated string or `NULL` if absent. */
SAFETENSORS_API const char *safetensors_metadata_get(
    const SafetensorsHandle *handle,
    const char *key
);

/* =========================================================================
 * Serialization (One-Shot & Incremental Builder)
 * ========================================================================= */

/**
 * Serializes an array of `SafetensorsTensorInput` tensors and optional metadata directly to `path`.
 */
SAFETENSORS_API SafetensorsStatus safetensors_save_to_file(
    const char *path,
    const SafetensorsTensorInput *tensors,
    size_t num_tensors,
    const SafetensorsMetadataEntry *metadata,
    size_t num_metadata
);

/**
 * Windows wide-character (`const uint16_t*` / `const wchar_t*`) variant of `safetensors_save_to_file`.
 */
SAFETENSORS_API SafetensorsStatus safetensors_save_to_file_w(
    const uint16_t *path_w,
    const SafetensorsTensorInput *tensors,
    size_t num_tensors,
    const SafetensorsMetadataEntry *metadata,
    size_t num_metadata
);

/**
 * Serializes an array of `SafetensorsTensorInput` tensors into a newly allocated buffer.
 * Free the returned buffer using `safetensors_free_buffer(*out_buffer, *out_len)`.
 */
SAFETENSORS_API SafetensorsStatus safetensors_save_to_buffer(
    const SafetensorsTensorInput *tensors,
    size_t num_tensors,
    const SafetensorsMetadataEntry *metadata,
    size_t num_metadata,
    uint8_t **out_buffer,
    size_t *out_len
);

/** Frees a byte buffer allocated by `safetensors_save_to_buffer` or `safetensors_serializer_save_buffer`. */
SAFETENSORS_API void safetensors_free_buffer(uint8_t *buffer, size_t len);

/** Creates a new incremental `SafetensorsSerializer` builder instance. */
SAFETENSORS_API SafetensorsSerializer *safetensors_serializer_new(void);

/** Adds a `__metadata__` key-value entry to `serializer`. */
SAFETENSORS_API SafetensorsStatus safetensors_serializer_add_metadata(
    SafetensorsSerializer *serializer,
    const char *key,
    const char *value
);

/** Adds a tensor (copying its bytes into the builder) to `serializer`. */
SAFETENSORS_API SafetensorsStatus safetensors_serializer_add_tensor(
    SafetensorsSerializer *serializer,
    const char *name,
    SafetensorsDtype dtype,
    const size_t *shape,
    size_t ndim,
    const void *data,
    size_t data_len
);

/** Writes all tensors and metadata accumulated in `serializer` to `path`. */
SAFETENSORS_API SafetensorsStatus safetensors_serializer_save_file(
    const SafetensorsSerializer *serializer,
    const char *path
);

/**
 * Serializes all tensors and metadata accumulated in `serializer` to an owned buffer.
 * Free the returned buffer using `safetensors_free_buffer(*out_buffer, *out_len)`.
 */
SAFETENSORS_API SafetensorsStatus safetensors_serializer_save_buffer(
    const SafetensorsSerializer *serializer,
    uint8_t **out_buffer,
    size_t *out_len
);

/** Frees a `SafetensorsSerializer` instance. */
SAFETENSORS_API void safetensors_serializer_free(SafetensorsSerializer *serializer);

#ifdef __cplusplus
}
#endif

#endif /* SAFETENSORS_H */
