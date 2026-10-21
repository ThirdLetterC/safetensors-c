cmake_minimum_required(VERSION 3.16)
project(safetensors_c VERSION 0.8.0 LANGUAGES C)

option(SAFETENSORS_BUILD_EXAMPLE "Build the example C executable" ON)
set(SAFETENSORS_RUST_TARGET "" CACHE STRING "Optional Rust target triple (e.g. x86_64-pc-windows-gnu)")

find_program(CARGO_EXECUTABLE cargo REQUIRED)

if(SAFETENSORS_RUST_TARGET STREQUAL "")
    set(CARGO_TARGET_ARGS "")
    set(CARGO_RELEASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
else()
    set(CARGO_TARGET_ARGS --target "${SAFETENSORS_RUST_TARGET}")
    set(CARGO_RELEASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/${SAFETENSORS_RUST_TARGET}/release")
endif()

set(DIST_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/dist/include")
set(DIST_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/dist/lib")
set(STAGED_HEADER "${DIST_INCLUDE_DIR}/safetensors.h")
set(STAGED_STATIC_A "${DIST_LIB_DIR}/libsafetensors_c.a")

if(WIN32 AND MSVC AND SAFETENSORS_RUST_TARGET STREQUAL "")
    set(CARGO_STATIC_OUTPUT "${CARGO_RELEASE_DIR}/safetensors_c.lib")
else()
    set(CARGO_STATIC_OUTPUT "${CARGO_RELEASE_DIR}/libsafetensors_c.a")
endif()

add_custom_command(
    OUTPUT "${STAGED_STATIC_A}" "${STAGED_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_INCLUDE_DIR}" "${DIST_LIB_DIR}"
    COMMAND "${CARGO_EXECUTABLE}" build --release ${CARGO_TARGET_ARGS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/include/safetensors.h"
            "${STAGED_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CARGO_STATIC_OUTPUT}"
            "${STAGED_STATIC_A}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building safetensors-c Rust static library and staging header + .a"
    VERBATIM
)

add_custom_target(safetensors_c_cargo ALL
    DEPENDS "${STAGED_STATIC_A}" "${STAGED_HEADER}"
)

add_library(safetensors_c STATIC IMPORTED GLOBAL)
add_dependencies(safetensors_c safetensors_c_cargo)
set_target_properties(safetensors_c PROPERTIES
    IMPORTED_LOCATION "${STAGED_STATIC_A}"
    INTERFACE_INCLUDE_DIRECTORIES "${DIST_INCLUDE_DIR}"
)

if(WIN32)
    target_link_libraries(safetensors_c INTERFACE ws2_32 userenv bcrypt ntdll advapi32)
elseif(APPLE)
    target_link_libraries(safetensors_c INTERFACE pthread m)
else()
    target_link_libraries(safetensors_c INTERFACE pthread dl m)
endif()

if(SAFETENSORS_BUILD_EXAMPLE)
    add_executable(safetensors_example examples/example.c)
    target_link_libraries(safetensors_example PRIVATE safetensors_c)
endif()
