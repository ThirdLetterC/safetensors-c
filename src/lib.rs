//! C FFI wrapper around the official `safetensors` Rust crate.
//!
//! Provides a panic-safe, zero-copy C API for reading, inspecting, and writing
//! `.safetensors` files and in-memory buffers across Linux, macOS, and Windows.

// The public functions expose a C ABI, so their pointer safety contract is
// documented for C callers rather than represented by Rust's `unsafe` API.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::manual_c_str_literals)]
#![allow(clippy::type_complexity)]

use libc::{c_char, c_void, size_t};
use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::{Dtype, SafeTensorError, SafeTensors, TensorView};
use safetensors::{serialize, serialize_to_file, View};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl Into<String>) {
    let s = msg.into();
    let c_str = CString::new(s.replace('\0', " ")).unwrap_or_else(|_| {
        CString::new("safetensors error (message contained interior NUL)").unwrap()
    });
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(c_str);
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

/// Status codes returned by all fallible `safetensors_*` C API functions.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum SafetensorsStatus {
    /// Operation completed without error.
    SAFETENSORS_OK = 0,
    /// A required pointer argument was NULL.
    SAFETENSORS_ERR_NULL_POINTER = 1,
    /// A string argument was not valid UTF-8 (or UTF-16 for `*_w` functions).
    SAFETENSORS_ERR_INVALID_UTF8 = 2,
    /// File I/O or memory-mapping failed.
    SAFETENSORS_ERR_IO = 3,
    /// Header or tensor buffer validation failed during deserialization.
    SAFETENSORS_ERR_DESERIALIZE = 4,
    /// Tensor validation or formatting failed during serialization.
    SAFETENSORS_ERR_SERIALIZE = 5,
    /// Requested tensor name or metadata key does not exist.
    SAFETENSORS_ERR_NOT_FOUND = 6,
    /// Requested tensor or metadata index is out of bounds.
    SAFETENSORS_ERR_INDEX_OUT_OF_BOUNDS = 7,
    /// Unrecognized or unsupported tensor element data type (`Dtype`).
    SAFETENSORS_ERR_INVALID_DTYPE = 8,
    /// Provided output buffer is too small.
    SAFETENSORS_ERR_BUFFER_TOO_SMALL = 9,
    /// An unexpected internal Rust panic was caught at the FFI boundary.
    SAFETENSORS_ERR_PANIC = 255,
}

/// Element data types supported by the `safetensors` format specification.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum SafetensorsDtype {
    /// Boolean (8-bit, 1 byte per element)
    SAFETENSORS_DTYPE_BOOL = 0,
    /// Unsigned 8-bit integer (`uint8_t`)
    SAFETENSORS_DTYPE_U8 = 1,
    /// Signed 8-bit integer (`int8_t`)
    SAFETENSORS_DTYPE_I8 = 2,
    /// 8-bit floating point (E5M2)
    SAFETENSORS_DTYPE_F8_E5M2 = 3,
    /// 8-bit floating point (E4M3)
    SAFETENSORS_DTYPE_F8_E4M3 = 4,
    /// 8-bit floating point (E8M0 microscaling exponent)
    SAFETENSORS_DTYPE_F8_E8M0 = 5,
    /// 8-bit floating point (AMD FNUZ E4M3)
    SAFETENSORS_DTYPE_F8_E4M3FNUZ = 6,
    /// 8-bit floating point (AMD FNUZ E5M2)
    SAFETENSORS_DTYPE_F8_E5M2FNUZ = 7,
    /// Signed 16-bit integer (`int16_t`)
    SAFETENSORS_DTYPE_I16 = 8,
    /// Unsigned 16-bit integer (`uint16_t`)
    SAFETENSORS_DTYPE_U16 = 9,
    /// IEEE 754 half-precision 16-bit float (`f16`)
    SAFETENSORS_DTYPE_F16 = 10,
    /// Brain floating point 16-bit (`bfloat16`)
    SAFETENSORS_DTYPE_BF16 = 11,
    /// Signed 32-bit integer (`int32_t`)
    SAFETENSORS_DTYPE_I32 = 12,
    /// Unsigned 32-bit integer (`uint32_t`)
    SAFETENSORS_DTYPE_U32 = 13,
    /// IEEE 754 single-precision 32-bit float (`float`)
    SAFETENSORS_DTYPE_F32 = 14,
    /// 64-bit complex float (two 32-bit floats: real, imag)
    SAFETENSORS_DTYPE_C64 = 15,
    /// IEEE 754 double-precision 64-bit float (`double`)
    SAFETENSORS_DTYPE_F64 = 16,
    /// Signed 64-bit integer (`int64_t`)
    SAFETENSORS_DTYPE_I64 = 17,
    /// Unsigned 64-bit integer (`uint64_t`)
    SAFETENSORS_DTYPE_U64 = 18,
    /// 4-bit microscaling float (`MXFP4`)
    SAFETENSORS_DTYPE_F4 = 19,
    /// 6-bit microscaling float (`MXFP6 E2M3`)
    SAFETENSORS_DTYPE_F6_E2M3 = 20,
    /// 6-bit microscaling float (`MXFP6 E3M2`)
    SAFETENSORS_DTYPE_F6_E3M2 = 21,
}

impl SafetensorsDtype {
    fn to_rust(self) -> Dtype {
        match self {
            SafetensorsDtype::SAFETENSORS_DTYPE_BOOL => Dtype::BOOL,
            SafetensorsDtype::SAFETENSORS_DTYPE_U8 => Dtype::U8,
            SafetensorsDtype::SAFETENSORS_DTYPE_I8 => Dtype::I8,
            SafetensorsDtype::SAFETENSORS_DTYPE_F8_E5M2 => Dtype::F8_E5M2,
            SafetensorsDtype::SAFETENSORS_DTYPE_F8_E4M3 => Dtype::F8_E4M3,
            SafetensorsDtype::SAFETENSORS_DTYPE_F8_E8M0 => Dtype::F8_E8M0,
            SafetensorsDtype::SAFETENSORS_DTYPE_F8_E4M3FNUZ => Dtype::F8_E4M3FNUZ,
            SafetensorsDtype::SAFETENSORS_DTYPE_F8_E5M2FNUZ => Dtype::F8_E5M2FNUZ,
            SafetensorsDtype::SAFETENSORS_DTYPE_I16 => Dtype::I16,
            SafetensorsDtype::SAFETENSORS_DTYPE_U16 => Dtype::U16,
            SafetensorsDtype::SAFETENSORS_DTYPE_F16 => Dtype::F16,
            SafetensorsDtype::SAFETENSORS_DTYPE_BF16 => Dtype::BF16,
            SafetensorsDtype::SAFETENSORS_DTYPE_I32 => Dtype::I32,
            SafetensorsDtype::SAFETENSORS_DTYPE_U32 => Dtype::U32,
            SafetensorsDtype::SAFETENSORS_DTYPE_F32 => Dtype::F32,
            SafetensorsDtype::SAFETENSORS_DTYPE_C64 => Dtype::C64,
            SafetensorsDtype::SAFETENSORS_DTYPE_F64 => Dtype::F64,
            SafetensorsDtype::SAFETENSORS_DTYPE_I64 => Dtype::I64,
            SafetensorsDtype::SAFETENSORS_DTYPE_U64 => Dtype::U64,
            SafetensorsDtype::SAFETENSORS_DTYPE_F4 => Dtype::F4,
            SafetensorsDtype::SAFETENSORS_DTYPE_F6_E2M3 => Dtype::F6_E2M3,
            SafetensorsDtype::SAFETENSORS_DTYPE_F6_E3M2 => Dtype::F6_E3M2,
        }
    }

    fn from_rust(dtype: Dtype) -> Result<Self, SafetensorsStatus> {
        #[allow(unreachable_patterns)]
        match dtype {
            Dtype::BOOL => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_BOOL),
            Dtype::U8 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_U8),
            Dtype::I8 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_I8),
            Dtype::F8_E5M2 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F8_E5M2),
            Dtype::F8_E4M3 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F8_E4M3),
            Dtype::F8_E8M0 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F8_E8M0),
            Dtype::F8_E4M3FNUZ => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F8_E4M3FNUZ),
            Dtype::F8_E5M2FNUZ => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F8_E5M2FNUZ),
            Dtype::I16 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_I16),
            Dtype::U16 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_U16),
            Dtype::F16 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F16),
            Dtype::BF16 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_BF16),
            Dtype::I32 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_I32),
            Dtype::U32 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_U32),
            Dtype::F32 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F32),
            Dtype::C64 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_C64),
            Dtype::F64 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F64),
            Dtype::I64 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_I64),
            Dtype::U64 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_U64),
            Dtype::F4 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F4),
            Dtype::F6_E2M3 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F6_E2M3),
            Dtype::F6_E3M2 => Ok(SafetensorsDtype::SAFETENSORS_DTYPE_F6_E3M2),
            _ => {
                set_last_error(format!("Unsupported safetensors Dtype: {:?}", dtype));
                Err(SafetensorsStatus::SAFETENSORS_ERR_INVALID_DTYPE)
            }
        }
    }

    fn static_name(self) -> &'static CStr {
        unsafe {
            match self {
                SafetensorsDtype::SAFETENSORS_DTYPE_BOOL => CStr::from_bytes_with_nul_unchecked(b"BOOL\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_U8 => CStr::from_bytes_with_nul_unchecked(b"U8\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_I8 => CStr::from_bytes_with_nul_unchecked(b"I8\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F8_E5M2 => CStr::from_bytes_with_nul_unchecked(b"F8_E5M2\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F8_E4M3 => CStr::from_bytes_with_nul_unchecked(b"F8_E4M3\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F8_E8M0 => CStr::from_bytes_with_nul_unchecked(b"F8_E8M0\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F8_E4M3FNUZ => CStr::from_bytes_with_nul_unchecked(b"F8_E4M3FNUZ\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F8_E5M2FNUZ => CStr::from_bytes_with_nul_unchecked(b"F8_E5M2FNUZ\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_I16 => CStr::from_bytes_with_nul_unchecked(b"I16\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_U16 => CStr::from_bytes_with_nul_unchecked(b"U16\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F16 => CStr::from_bytes_with_nul_unchecked(b"F16\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_BF16 => CStr::from_bytes_with_nul_unchecked(b"BF16\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_I32 => CStr::from_bytes_with_nul_unchecked(b"I32\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_U32 => CStr::from_bytes_with_nul_unchecked(b"U32\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F32 => CStr::from_bytes_with_nul_unchecked(b"F32\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_C64 => CStr::from_bytes_with_nul_unchecked(b"C64\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F64 => CStr::from_bytes_with_nul_unchecked(b"F64\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_I64 => CStr::from_bytes_with_nul_unchecked(b"I64\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_U64 => CStr::from_bytes_with_nul_unchecked(b"U64\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F4 => CStr::from_bytes_with_nul_unchecked(b"F4\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F6_E2M3 => CStr::from_bytes_with_nul_unchecked(b"F6_E2M3\0"),
                SafetensorsDtype::SAFETENSORS_DTYPE_F6_E3M2 => CStr::from_bytes_with_nul_unchecked(b"F6_E3M2\0"),
            }
        }
    }
}

/// Zero-copy view of a deserialized tensor returned to C callers.
///
/// All pointers (`name`, `shape`, `data`) are owned by the parent `SafetensorsHandle`
/// and remain valid until `safetensors_free(handle)` is called.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SafetensorsTensorView {
    /// Null-terminated UTF-8 tensor name.
    pub name: *const c_char,
    /// Element data type (`SafetensorsDtype`).
    pub dtype: SafetensorsDtype,
    /// Number of dimensions (rank). `0` for scalar tensors.
    pub ndim: size_t,
    /// Pointer to array of `ndim` dimension sizes (`NULL` only if `ndim == 0`).
    pub shape: *const size_t,
    /// Total number of elements in the tensor (product of `shape[0..ndim]`).
    pub num_elements: size_t,
    /// Zero-copy pointer to the raw little-endian tensor bytes.
    pub data: *const c_void,
    /// Length of the tensor buffer in bytes.
    pub data_len: size_t,
}

/// Tensor descriptor passed by C callers when serializing tensors to a file or buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SafetensorsTensorInput {
    /// Null-terminated UTF-8 tensor name.
    pub name: *const c_char,
    /// Element data type (`SafetensorsDtype`).
    pub dtype: SafetensorsDtype,
    /// Number of dimensions (rank).
    pub ndim: size_t,
    /// Pointer to array of `ndim` dimension sizes (may be `NULL` if `ndim == 0`).
    pub shape: *const size_t,
    /// Pointer to raw little-endian tensor byte buffer.
    pub data: *const c_void,
    /// Length of `data` in bytes.
    pub data_len: size_t,
}

/// Key-value metadata entry for the `__metadata__` JSON header map.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SafetensorsMetadataEntry {
    /// Null-terminated UTF-8 key string.
    pub key: *const c_char,
    /// Null-terminated UTF-8 value string.
    pub value: *const c_char,
}

enum BackingStorage {
    Mmap(Mmap),
    Owned(Box<[u8]>),
    Borrowed { ptr: *const u8, len: usize },
}

unsafe impl Send for BackingStorage {}
unsafe impl Sync for BackingStorage {}

impl BackingStorage {
    fn as_slice(&self) -> &[u8] {
        match self {
            BackingStorage::Mmap(mmap) => &mmap[..],
            BackingStorage::Owned(buf) => &buf[..],
            BackingStorage::Borrowed { ptr, len } => unsafe { slice::from_raw_parts(*ptr, *len) },
        }
    }
}

struct TensorEntry {
    name: CString,
    dtype: SafetensorsDtype,
    shape: Vec<size_t>,
    num_elements: size_t,
    byte_offset: usize,
    data_len: usize,
}

struct MetadataPair {
    key: CString,
    value: CString,
}

/// Opaque handle representing a loaded `.safetensors` file or buffer.
pub struct SafetensorsHandle {
    storage: BackingStorage,
    tensors: Vec<TensorEntry>,
    tensor_index: HashMap<String, usize>,
    metadata: Vec<MetadataPair>,
    metadata_index: HashMap<String, usize>,
}

impl SafetensorsHandle {
    fn from_storage(storage: BackingStorage) -> Result<Self, SafetensorsStatus> {
        let buffer = storage.as_slice();

        let (_header_size, meta) = SafeTensors::read_metadata(buffer).map_err(|err| {
            set_last_error(format!("SafeTensors::read_metadata failed: {}", err));
            SafetensorsStatus::SAFETENSORS_ERR_DESERIALIZE
        })?;

        let st = SafeTensors::deserialize(buffer).map_err(|err| {
            set_last_error(format!("SafeTensors::deserialize failed: {}", err));
            SafetensorsStatus::SAFETENSORS_ERR_DESERIALIZE
        })?;

        let ordered_keys = meta.offset_keys();
        let mut tensors = Vec::with_capacity(ordered_keys.len());
        let mut tensor_index = HashMap::with_capacity(ordered_keys.len());
        let base_ptr = buffer.as_ptr() as usize;

        for (idx, key) in ordered_keys.into_iter().enumerate() {
            let view = st.tensor(&key).map_err(|err| {
                set_last_error(format!("Failed to resolve tensor '{}': {}", key, err));
                SafetensorsStatus::SAFETENSORS_ERR_DESERIALIZE
            })?;

            let dtype = SafetensorsDtype::from_rust(view.dtype())?;
            let shape: Vec<size_t> = view.shape().to_vec();
            let num_elements: size_t = shape.iter().copied().product();
            let slice_data = view.data();
            let byte_offset = (slice_data.as_ptr() as usize).saturating_sub(base_ptr);
            let data_len = view.data_len();

            let c_name = CString::new(key.as_str()).map_err(|_| {
                set_last_error("Tensor name contains interior NUL byte");
                SafetensorsStatus::SAFETENSORS_ERR_INVALID_UTF8
            })?;

            tensor_index.insert(key, idx);
            tensors.push(TensorEntry {
                name: c_name,
                dtype,
                shape,
                num_elements,
                byte_offset,
                data_len,
            });
        }

        let mut metadata = Vec::new();
        let mut metadata_index = HashMap::new();
        if let Some(map) = meta.metadata() {
            let mut pairs: Vec<(&String, &String)> = map.iter().collect();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            metadata.reserve(pairs.len());
            for (idx, (k, v)) in pairs.into_iter().enumerate() {
                let c_key = CString::new(k.as_str()).map_err(|_| {
                    set_last_error("Metadata key contains interior NUL byte");
                    SafetensorsStatus::SAFETENSORS_ERR_INVALID_UTF8
                })?;
                let c_val = CString::new(v.as_str()).map_err(|_| {
                    set_last_error("Metadata value contains interior NUL byte");
                    SafetensorsStatus::SAFETENSORS_ERR_INVALID_UTF8
                })?;
                metadata_index.insert(k.clone(), idx);
                metadata.push(MetadataPair {
                    key: c_key,
                    value: c_val,
                });
            }
        }

        Ok(SafetensorsHandle {
            storage,
            tensors,
            tensor_index,
            metadata,
            metadata_index,
        })
    }

    fn make_tensor_view(&self, entry: &TensorEntry) -> SafetensorsTensorView {
        let base_ptr = self.storage.as_slice().as_ptr();
        let data_ptr = unsafe { base_ptr.add(entry.byte_offset) as *const c_void };
        let shape_ptr = if entry.shape.is_empty() {
            ptr::null()
        } else {
            entry.shape.as_ptr()
        };

        SafetensorsTensorView {
            name: entry.name.as_ptr(),
            dtype: entry.dtype,
            ndim: entry.shape.len(),
            shape: shape_ptr,
            num_elements: entry.num_elements,
            data: data_ptr,
            data_len: entry.data_len,
        }
    }
}

struct OwnedTensorSpec {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Opaque builder handle for incrementally constructing and saving `.safetensors` files.
#[derive(Default)]
pub struct SafetensorsSerializer {
    tensors: Vec<OwnedTensorSpec>,
    metadata: HashMap<String, String>,
}

fn cstr_to_str<'a>(ptr: *const c_char, label: &str) -> Result<&'a str, SafetensorsStatus> {
    if ptr.is_null() {
        set_last_error(format!("{} pointer is NULL", label));
        return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|err| {
        set_last_error(format!("{} is not valid UTF-8: {}", label, err));
        SafetensorsStatus::SAFETENSORS_ERR_INVALID_UTF8
    })
}

fn wstr_to_pathbuf(ptr: *const u16, label: &str) -> Result<PathBuf, SafetensorsStatus> {
    if ptr.is_null() {
        set_last_error(format!("{} wide string pointer is NULL", label));
        return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
    }
    let mut len = 0usize;
    unsafe {
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let slice = slice::from_raw_parts(ptr, len);
        #[cfg(windows)]
        {
            use std::ffi::OsString;
            use std::os::windows::ffi::OsStringExt;
            Ok(PathBuf::from(OsString::from_wide(slice)))
        }
        #[cfg(not(windows))]
        {
            let s = String::from_utf16(slice).map_err(|err| {
                set_last_error(format!("{} is not valid UTF-16: {}", label, err));
                SafetensorsStatus::SAFETENSORS_ERR_INVALID_UTF8
            })?;
            Ok(PathBuf::from(s))
        }
    }
}

fn ffi_guard<F>(f: F) -> SafetensorsStatus
where
    F: FnOnce() -> Result<(), SafetensorsStatus>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            clear_last_error();
            SafetensorsStatus::SAFETENSORS_OK
        }
        Ok(Err(status)) => status,
        Err(_) => {
            set_last_error("Internal panic caught across FFI boundary");
            SafetensorsStatus::SAFETENSORS_ERR_PANIC
        }
    }
}

// ===========================================================================
// Version & Error Reporting Functions
// ===========================================================================

/// Returns the static null-terminated version string of `safetensors-c` (e.g. `"0.8.0"`).
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Returns the most recent error message on the calling thread, or `NULL` if no error occurred.
///
/// The returned pointer is valid until the next `safetensors_*` call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| match *slot.borrow() {
        Some(ref cstr) => cstr.as_ptr(),
        None => ptr::null(),
    })
}

/// Returns a human-readable static string describing a `SafetensorsStatus` code.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_status_str(status: SafetensorsStatus) -> *const c_char {
    let bytes: &'static [u8] = match status {
        SafetensorsStatus::SAFETENSORS_OK => b"OK\0",
        SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER => b"Null pointer argument\0",
        SafetensorsStatus::SAFETENSORS_ERR_INVALID_UTF8 => b"Invalid UTF-8/UTF-16 string\0",
        SafetensorsStatus::SAFETENSORS_ERR_IO => b"I/O or memory-map error\0",
        SafetensorsStatus::SAFETENSORS_ERR_DESERIALIZE => b"Safetensors deserialization error\0",
        SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE => b"Safetensors serialization error\0",
        SafetensorsStatus::SAFETENSORS_ERR_NOT_FOUND => b"Requested tensor or metadata key not found\0",
        SafetensorsStatus::SAFETENSORS_ERR_INDEX_OUT_OF_BOUNDS => b"Index out of bounds\0",
        SafetensorsStatus::SAFETENSORS_ERR_INVALID_DTYPE => b"Invalid or unsupported tensor dtype\0",
        SafetensorsStatus::SAFETENSORS_ERR_BUFFER_TOO_SMALL => b"Output buffer too small\0",
        SafetensorsStatus::SAFETENSORS_ERR_PANIC => b"Internal Rust panic caught at FFI boundary\0",
    };
    bytes.as_ptr() as *const c_char
}

// ===========================================================================
// Dtype Utility Functions
// ===========================================================================

/// Returns the size in bits of a single element of `dtype` (e.g., `32` for `F32`, `4` for `F4`).
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_dtype_bitsize(dtype: SafetensorsDtype) -> size_t {
    dtype.to_rust().bitsize()
}

/// Returns the size in bytes of a single element of `dtype`, or `0` for sub-byte types (`F4`, `F6_*`).
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_dtype_size(dtype: SafetensorsDtype) -> size_t {
    let bits = dtype.to_rust().bitsize();
    if bits % 8 == 0 {
        bits / 8
    } else {
        0
    }
}

/// Returns the canonical null-terminated string name of `dtype` (e.g. `"F32"`, `"BF16"`).
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_dtype_name(dtype: SafetensorsDtype) -> *const c_char {
    dtype.static_name().as_ptr()
}

/// Parses a safetensors dtype string (e.g. `"F32"`, `"BF16"`, `"I64"`) into `out_dtype`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_dtype_from_str(
    name: *const c_char,
    out_dtype: *mut SafetensorsDtype,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if out_dtype.is_null() {
            set_last_error("out_dtype pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let s = cstr_to_str(name, "dtype name")?;
        let parsed = match s {
            "BOOL" => Dtype::BOOL,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "F8_E5M2" => Dtype::F8_E5M2,
            "F8_E4M3" => Dtype::F8_E4M3,
            "F8_E8M0" => Dtype::F8_E8M0,
            "F8_E4M3FNUZ" => Dtype::F8_E4M3FNUZ,
            "F8_E5M2FNUZ" => Dtype::F8_E5M2FNUZ,
            "I16" => Dtype::I16,
            "U16" => Dtype::U16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I32" => Dtype::I32,
            "U32" => Dtype::U32,
            "F32" => Dtype::F32,
            "C64" => Dtype::C64,
            "F64" => Dtype::F64,
            "I64" => Dtype::I64,
            "U64" => Dtype::U64,
            "F4" => Dtype::F4,
            "F6_E2M3" => Dtype::F6_E2M3,
            "F6_E3M2" => Dtype::F6_E3M2,
            _ => {
                set_last_error(format!("Unrecognized safetensors dtype string: '{}'", s));
                return Err(SafetensorsStatus::SAFETENSORS_ERR_INVALID_DTYPE);
            }
        };
        let c_dtype = SafetensorsDtype::from_rust(parsed)?;
        unsafe {
            *out_dtype = c_dtype;
        }
        Ok(())
    })
}

// ===========================================================================
// Deserialization & File Loading Functions
// ===========================================================================

fn open_path_mmap(path: &Path) -> Result<SafetensorsHandle, SafetensorsStatus> {
    let file = File::open(path).map_err(|err| {
        set_last_error(format!("Failed to open file '{}': {}", path.display(), err));
        SafetensorsStatus::SAFETENSORS_ERR_IO
    })?;
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|err| {
        set_last_error(format!(
            "Failed to memory-map file '{}': {}",
            path.display(),
            err
        ));
        SafetensorsStatus::SAFETENSORS_ERR_IO
    })?;
    SafetensorsHandle::from_storage(BackingStorage::Mmap(mmap))
}

fn open_path_buffered(path: &Path) -> Result<SafetensorsHandle, SafetensorsStatus> {
    let bytes = std::fs::read(path).map_err(|err| {
        set_last_error(format!("Failed to read file '{}': {}", path.display(), err));
        SafetensorsStatus::SAFETENSORS_ERR_IO
    })?;
    SafetensorsHandle::from_storage(BackingStorage::Owned(bytes.into_boxed_slice()))
}

/// Opens a `.safetensors` file using cross-platform OS memory mapping (`mmap` on Unix,
/// `CreateFileMappingW` / `MapViewOfFile` on Windows) for zero-copy tensor access.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_open_file(
    path: *const c_char,
    out_handle: *mut *mut SafetensorsHandle,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if out_handle.is_null() {
            set_last_error("out_handle pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        unsafe {
            *out_handle = ptr::null_mut();
        }
        let path_str = cstr_to_str(path, "path")?;
        let handle = open_path_mmap(Path::new(path_str))?;
        unsafe {
            *out_handle = Box::into_raw(Box::new(handle));
        }
        Ok(())
    })
}

/// Windows wide-string (`const uint16_t*` / `LPCWSTR`) variant of `safetensors_open_file`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_open_file_w(
    path_w: *const u16,
    out_handle: *mut *mut SafetensorsHandle,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if out_handle.is_null() {
            set_last_error("out_handle pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        unsafe {
            *out_handle = ptr::null_mut();
        }
        let path_buf = wstr_to_pathbuf(path_w, "path_w")?;
        let handle = open_path_mmap(&path_buf)?;
        unsafe {
            *out_handle = Box::into_raw(Box::new(handle));
        }
        Ok(())
    })
}

/// Opens a `.safetensors` file by reading the entire file into a heap-allocated buffer.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_open_file_buffered(
    path: *const c_char,
    out_handle: *mut *mut SafetensorsHandle,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if out_handle.is_null() {
            set_last_error("out_handle pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        unsafe {
            *out_handle = ptr::null_mut();
        }
        let path_str = cstr_to_str(path, "path")?;
        let handle = open_path_buffered(Path::new(path_str))?;
        unsafe {
            *out_handle = Box::into_raw(Box::new(handle));
        }
        Ok(())
    })
}

/// Deserializes a `.safetensors` byte buffer by copying `buffer[0..len]` into the handle.
///
/// The caller may free or reuse `buffer` immediately after this call returns.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_deserialize_copy(
    buffer: *const u8,
    len: size_t,
    out_handle: *mut *mut SafetensorsHandle,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if out_handle.is_null() {
            set_last_error("out_handle pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        unsafe {
            *out_handle = ptr::null_mut();
        }
        if buffer.is_null() {
            set_last_error("buffer pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let slice = unsafe { slice::from_raw_parts(buffer, len) };
        let owned = slice.to_vec().into_boxed_slice();
        let handle = SafetensorsHandle::from_storage(BackingStorage::Owned(owned))?;
        unsafe {
            *out_handle = Box::into_raw(Box::new(handle));
        }
        Ok(())
    })
}

/// Deserializes a `.safetensors` byte buffer in zero-copy mode without copying `buffer`.
///
/// # Safety Contract for C Callers
/// The caller MUST keep `buffer` alive and unmodified for the entire lifetime of `out_handle`
/// until `safetensors_free(out_handle)` is called.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_deserialize_view(
    buffer: *const u8,
    len: size_t,
    out_handle: *mut *mut SafetensorsHandle,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if out_handle.is_null() {
            set_last_error("out_handle pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        unsafe {
            *out_handle = ptr::null_mut();
        }
        if buffer.is_null() {
            set_last_error("buffer pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let handle = SafetensorsHandle::from_storage(BackingStorage::Borrowed {
            ptr: buffer,
            len,
        })?;
        unsafe {
            *out_handle = Box::into_raw(Box::new(handle));
        }
        Ok(())
    })
}

/// Releases all resources (including any memory map or owned buffer) held by `handle`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_free(handle: *mut SafetensorsHandle) {
    if !handle.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(handle));
        }));
    }
}

// ===========================================================================
// Tensor Inspection Functions
// ===========================================================================

/// Returns the number of tensors stored in `handle` (or `0` if `handle` is `NULL`).
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_num_tensors(handle: *const SafetensorsHandle) -> size_t {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).tensors.len() }
}

/// Returns the null-terminated name of the tensor at `index`, or `NULL` if out of bounds.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_tensor_name(
    handle: *const SafetensorsHandle,
    index: size_t,
) -> *const c_char {
    if handle.is_null() {
        return ptr::null();
    }
    let h = unsafe { &*handle };
    match h.tensors.get(index) {
        Some(entry) => entry.name.as_ptr(),
        None => ptr::null(),
    }
}

/// Retrieves a zero-copy `SafetensorsTensorView` for the tensor at `index`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_get_tensor_by_index(
    handle: *const SafetensorsHandle,
    index: size_t,
    out_view: *mut SafetensorsTensorView,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if handle.is_null() {
            set_last_error("handle pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        if out_view.is_null() {
            set_last_error("out_view pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let h = unsafe { &*handle };
        let entry = h.tensors.get(index).ok_or_else(|| {
            set_last_error(format!(
                "Tensor index {} out of bounds (num_tensors = {})",
                index,
                h.tensors.len()
            ));
            SafetensorsStatus::SAFETENSORS_ERR_INDEX_OUT_OF_BOUNDS
        })?;
        unsafe {
            *out_view = h.make_tensor_view(entry);
        }
        Ok(())
    })
}

/// Retrieves a zero-copy `SafetensorsTensorView` for the tensor with the given UTF-8 `name`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_get_tensor_by_name(
    handle: *const SafetensorsHandle,
    name: *const c_char,
    out_view: *mut SafetensorsTensorView,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if handle.is_null() {
            set_last_error("handle pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        if out_view.is_null() {
            set_last_error("out_view pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let name_str = cstr_to_str(name, "tensor name")?;
        let h = unsafe { &*handle };
        let &idx = h.tensor_index.get(name_str).ok_or_else(|| {
            set_last_error(format!("Tensor '{}' not found in safetensors handle", name_str));
            SafetensorsStatus::SAFETENSORS_ERR_NOT_FOUND
        })?;
        let entry = &h.tensors[idx];
        unsafe {
            *out_view = h.make_tensor_view(entry);
        }
        Ok(())
    })
}

// ===========================================================================
// Header Metadata (__metadata__) Inspection Functions
// ===========================================================================

/// Returns the number of custom key-value pairs in the `__metadata__` header section.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_num_metadata(handle: *const SafetensorsHandle) -> size_t {
    if handle.is_null() {
        return 0;
    }
    unsafe { (*handle).metadata.len() }
}

/// Returns the metadata key at `index`, or `NULL` if `index` is out of bounds.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_metadata_key(
    handle: *const SafetensorsHandle,
    index: size_t,
) -> *const c_char {
    if handle.is_null() {
        return ptr::null();
    }
    let h = unsafe { &*handle };
    match h.metadata.get(index) {
        Some(pair) => pair.key.as_ptr(),
        None => ptr::null(),
    }
}

/// Returns the metadata value at `index`, or `NULL` if `index` is out of bounds.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_metadata_value(
    handle: *const SafetensorsHandle,
    index: size_t,
) -> *const c_char {
    if handle.is_null() {
        return ptr::null();
    }
    let h = unsafe { &*handle };
    match h.metadata.get(index) {
        Some(pair) => pair.value.as_ptr(),
        None => ptr::null(),
    }
}

/// Looks up a `__metadata__` value by `key`, returning a null-terminated string or `NULL` if absent.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_metadata_get(
    handle: *const SafetensorsHandle,
    key: *const c_char,
) -> *const c_char {
    if handle.is_null() || key.is_null() {
        return ptr::null();
    }
    let Ok(key_str) = (unsafe { CStr::from_ptr(key) }).to_str() else {
        return ptr::null();
    };
    let h = unsafe { &*handle };
    match h.metadata_index.get(key_str) {
        Some(&idx) => h.metadata[idx].value.as_ptr(),
        None => ptr::null(),
    }
}

// ===========================================================================
// Serialization Functions (One-Shot & Builder API)
// ===========================================================================

fn build_views_and_metadata<'a>(
    tensors: *const SafetensorsTensorInput,
    num_tensors: size_t,
    metadata: *const SafetensorsMetadataEntry,
    num_metadata: size_t,
) -> Result<
    (
        Vec<(&'a str, TensorView<'a>)>,
        Option<HashMap<String, String>>,
    ),
    SafetensorsStatus,
> {
    if num_tensors > 0 && tensors.is_null() {
        set_last_error("tensors array pointer is NULL while num_tensors > 0");
        return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
    }
    if num_metadata > 0 && metadata.is_null() {
        set_last_error("metadata array pointer is NULL while num_metadata > 0");
        return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
    }

    let tensor_slice = if num_tensors == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(tensors, num_tensors) }
    };

    let mut views = Vec::with_capacity(num_tensors);
    for (i, item) in tensor_slice.iter().enumerate() {
        let name = cstr_to_str(item.name, &format!("tensors[{}].name", i))?;
        if item.ndim > 0 && item.shape.is_null() {
            set_last_error(format!("tensors[{}] ('{}') shape pointer is NULL", i, name));
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        if item.data_len > 0 && item.data.is_null() {
            set_last_error(format!("tensors[{}] ('{}') data pointer is NULL", i, name));
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }

        let shape_vec: Vec<usize> = if item.ndim == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(item.shape, item.ndim).to_vec() }
        };

        let data_bytes: &'a [u8] = if item.data_len == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(item.data as *const u8, item.data_len) }
        };

        let view = TensorView::new(item.dtype.to_rust(), shape_vec, data_bytes).map_err(
            |err: SafeTensorError| {
                set_last_error(format!("Invalid tensor '{}': {}", name, err));
                SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
            },
        )?;
        views.push((name, view));
    }

    let meta_opt = if num_metadata == 0 {
        None
    } else {
        let meta_slice = unsafe { slice::from_raw_parts(metadata, num_metadata) };
        let mut map = HashMap::with_capacity(num_metadata);
        for (i, entry) in meta_slice.iter().enumerate() {
            let k = cstr_to_str(entry.key, &format!("metadata[{}].key", i))?;
            let v = cstr_to_str(entry.value, &format!("metadata[{}].value", i))?;
            map.insert(k.to_owned(), v.to_owned());
        }
        Some(map)
    };

    Ok((views, meta_opt))
}

/// Serializes an array of `SafetensorsTensorInput` tensors and optional metadata directly to `path`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_save_to_file(
    path: *const c_char,
    tensors: *const SafetensorsTensorInput,
    num_tensors: size_t,
    metadata: *const SafetensorsMetadataEntry,
    num_metadata: size_t,
) -> SafetensorsStatus {
    ffi_guard(|| {
        let path_str = cstr_to_str(path, "path")?;
        let (views, meta_opt) =
            build_views_and_metadata(tensors, num_tensors, metadata, num_metadata)?;
        serialize_to_file(views, meta_opt, Path::new(path_str)).map_err(|err| {
            set_last_error(format!(
                "safetensors::serialize_to_file('{}') failed: {}",
                path_str, err
            ));
            SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
        })
    })
}

/// Windows wide-string (`const uint16_t*` / `LPCWSTR`) variant of `safetensors_save_to_file`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_save_to_file_w(
    path_w: *const u16,
    tensors: *const SafetensorsTensorInput,
    num_tensors: size_t,
    metadata: *const SafetensorsMetadataEntry,
    num_metadata: size_t,
) -> SafetensorsStatus {
    ffi_guard(|| {
        let path_buf = wstr_to_pathbuf(path_w, "path_w")?;
        let (views, meta_opt) =
            build_views_and_metadata(tensors, num_tensors, metadata, num_metadata)?;
        serialize_to_file(views, meta_opt, &path_buf).map_err(|err| {
            set_last_error(format!(
                "safetensors::serialize_to_file('{}') failed: {}",
                path_buf.display(),
                err
            ));
            SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
        })
    })
}

/// Serializes an array of `SafetensorsTensorInput` tensors into a newly allocated byte buffer.
///
/// Free the returned buffer using `safetensors_free_buffer(*out_buffer, *out_len)`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_save_to_buffer(
    tensors: *const SafetensorsTensorInput,
    num_tensors: size_t,
    metadata: *const SafetensorsMetadataEntry,
    num_metadata: size_t,
    out_buffer: *mut *mut u8,
    out_len: *mut size_t,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if out_buffer.is_null() || out_len.is_null() {
            set_last_error("out_buffer or out_len pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        unsafe {
            *out_buffer = ptr::null_mut();
            *out_len = 0;
        }
        let (views, meta_opt) =
            build_views_and_metadata(tensors, num_tensors, metadata, num_metadata)?;
        let bytes = serialize(views, meta_opt).map_err(|err| {
            set_last_error(format!("safetensors::serialize failed: {}", err));
            SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
        })?;
        let mut boxed = bytes.into_boxed_slice();
        let len = boxed.len();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        unsafe {
            *out_buffer = ptr;
            *out_len = len;
        }
        Ok(())
    })
}

/// Frees a byte buffer previously allocated by `safetensors_save_to_buffer` or
/// `safetensors_serializer_save_buffer`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_free_buffer(buffer: *mut u8, len: size_t) {
    if !buffer.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            let slice_ptr = ptr::slice_from_raw_parts_mut(buffer, len);
            drop(Box::from_raw(slice_ptr));
        }));
    }
}

/// Creates a new incremental `SafetensorsSerializer` builder instance.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_serializer_new() -> *mut SafetensorsSerializer {
    Box::into_raw(Box::new(SafetensorsSerializer::default()))
}

/// Adds a `__metadata__` key-value entry to a `SafetensorsSerializer` builder.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_serializer_add_metadata(
    serializer: *mut SafetensorsSerializer,
    key: *const c_char,
    value: *const c_char,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if serializer.is_null() {
            set_last_error("serializer pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let k = cstr_to_str(key, "metadata key")?;
        let v = cstr_to_str(value, "metadata value")?;
        let ser = unsafe { &mut *serializer };
        ser.metadata.insert(k.to_owned(), v.to_owned());
        Ok(())
    })
}

/// Adds a tensor (copying its bytes into the builder) to a `SafetensorsSerializer`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_serializer_add_tensor(
    serializer: *mut SafetensorsSerializer,
    name: *const c_char,
    dtype: SafetensorsDtype,
    shape: *const size_t,
    ndim: size_t,
    data: *const c_void,
    data_len: size_t,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if serializer.is_null() {
            set_last_error("serializer pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let name_str = cstr_to_str(name, "tensor name")?;
        if ndim > 0 && shape.is_null() {
            set_last_error("shape pointer is NULL while ndim > 0");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        if data_len > 0 && data.is_null() {
            set_last_error("data pointer is NULL while data_len > 0");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }

        let shape_vec: Vec<usize> = if ndim == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(shape, ndim).to_vec() }
        };
        let bytes: Vec<u8> = if data_len == 0 {
            Vec::new()
        } else {
            unsafe { slice::from_raw_parts(data as *const u8, data_len).to_vec() }
        };

        let rust_dtype = dtype.to_rust();
        // Validate eagerly using TensorView::new
        let _ = TensorView::new(rust_dtype, shape_vec.clone(), &bytes).map_err(|err| {
            set_last_error(format!("Invalid tensor '{}': {}", name_str, err));
            SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
        })?;

        let ser = unsafe { &mut *serializer };
        ser.tensors.push(OwnedTensorSpec {
            name: name_str.to_owned(),
            dtype: rust_dtype,
            shape: shape_vec,
            data: bytes,
        });
        Ok(())
    })
}

/// Writes all tensors and metadata accumulated in `serializer` to `path`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_serializer_save_file(
    serializer: *const SafetensorsSerializer,
    path: *const c_char,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if serializer.is_null() {
            set_last_error("serializer pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        let path_str = cstr_to_str(path, "path")?;
        let ser = unsafe { &*serializer };

        let mut views = Vec::with_capacity(ser.tensors.len());
        for spec in &ser.tensors {
            let v = TensorView::new(spec.dtype, spec.shape.clone(), &spec.data).map_err(|err| {
                set_last_error(format!("Invalid tensor '{}': {}", spec.name, err));
                SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
            })?;
            views.push((spec.name.as_str(), v));
        }

        let meta_opt = if ser.metadata.is_empty() {
            None
        } else {
            Some(ser.metadata.clone())
        };

        serialize_to_file(views, meta_opt, Path::new(path_str)).map_err(|err| {
            set_last_error(format!(
                "safetensors::serialize_to_file('{}') failed: {}",
                path_str, err
            ));
            SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
        })
    })
}

/// Serializes all tensors and metadata in `serializer` to a newly allocated byte buffer.
///
/// Free the returned buffer with `safetensors_free_buffer(*out_buffer, *out_len)`.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_serializer_save_buffer(
    serializer: *const SafetensorsSerializer,
    out_buffer: *mut *mut u8,
    out_len: *mut size_t,
) -> SafetensorsStatus {
    ffi_guard(|| {
        if serializer.is_null() || out_buffer.is_null() || out_len.is_null() {
            set_last_error("serializer, out_buffer, or out_len pointer is NULL");
            return Err(SafetensorsStatus::SAFETENSORS_ERR_NULL_POINTER);
        }
        unsafe {
            *out_buffer = ptr::null_mut();
            *out_len = 0;
        }
        let ser = unsafe { &*serializer };

        let mut views = Vec::with_capacity(ser.tensors.len());
        for spec in &ser.tensors {
            let v = TensorView::new(spec.dtype, spec.shape.clone(), &spec.data).map_err(|err| {
                set_last_error(format!("Invalid tensor '{}': {}", spec.name, err));
                SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
            })?;
            views.push((spec.name.as_str(), v));
        }

        let meta_opt = if ser.metadata.is_empty() {
            None
        } else {
            Some(ser.metadata.clone())
        };

        let bytes = serialize(views, meta_opt).map_err(|err| {
            set_last_error(format!("safetensors::serialize failed: {}", err));
            SafetensorsStatus::SAFETENSORS_ERR_SERIALIZE
        })?;
        let mut boxed = bytes.into_boxed_slice();
        let len = boxed.len();
        let ptr = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        unsafe {
            *out_buffer = ptr;
            *out_len = len;
        }
        Ok(())
    })
}

/// Frees a `SafetensorsSerializer` instance.
#[unsafe(no_mangle)]
pub extern "C" fn safetensors_serializer_free(serializer: *mut SafetensorsSerializer) {
    if !serializer.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(serializer));
        }));
    }
}
