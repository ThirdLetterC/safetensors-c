/**
 * @file example.c
 * @brief Complete C example demonstrating serialization, file saving, memory-mapped
 *        loading, metadata inspection, and zero-copy tensor reading with `safetensors.h`.
 */

#include "safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check_status(SafetensorsStatus status, const char *context) {
    if (status != SAFETENSORS_OK) {
        const char *detail = safetensors_last_error_message();
        fprintf(stderr, "[ERROR] %s: %s (%s)\n",
                context,
                safetensors_status_str(status),
                detail ? detail : "no additional details");
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    printf("safetensors-c version: %s\n", safetensors_version());

    /* 1. Prepare sample tensors (2x3 float32 weight matrix + 3-element float32 bias) */
    const float weights[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const size_t weight_shape[2] = {2, 3};

    const float bias[3] = {0.1f, 0.2f, 0.3f};
    const size_t bias_shape[1] = {3};

    const SafetensorsTensorInput inputs[2] = {
        {
            .name = "layer.weight",
            .dtype = SAFETENSORS_DTYPE_F32,
            .ndim = 2,
            .shape = weight_shape,
            .data = weights,
            .data_len = sizeof(weights)
        },
        {
            .name = "layer.bias",
            .dtype = SAFETENSORS_DTYPE_F32,
            .ndim = 1,
            .shape = bias_shape,
            .data = bias,
            .data_len = sizeof(bias)
        }
    };

    const SafetensorsMetadataEntry metadata[2] = {
        { .key = "format", .value = "pt" },
        { .key = "producer", .value = "safetensors-c-example" }
    };

    const char *file_path = "example_model.safetensors";

    /* 2. Save tensors and metadata to disk */
    SafetensorsStatus st = safetensors_save_to_file(
        file_path,
        inputs, 2,
        metadata, 2
    );
    check_status(st, "safetensors_save_to_file");
    printf("Saved '%s' with 2 tensors and 2 metadata entries.\n", file_path);

    /* 3. Memory-map the .safetensors file for zero-copy reading */
    SafetensorsHandle *handle = NULL;
    st = safetensors_open_file(file_path, &handle);
    check_status(st, "safetensors_open_file");

    /* 4. Inspect header __metadata__ */
    size_t num_meta = safetensors_num_metadata(handle);
    printf("Header metadata (%zu entries):\n", num_meta);
    for (size_t i = 0; i < num_meta; ++i) {
        printf("  %s = %s\n",
               safetensors_metadata_key(handle, i),
               safetensors_metadata_value(handle, i));
    }

    /* 5. Look up a tensor by name and read its zero-copy data pointer */
    SafetensorsTensorView view;
    st = safetensors_get_tensor_by_name(handle, "layer.weight", &view);
    check_status(st, "safetensors_get_tensor_by_name");

    printf("Tensor '%s': dtype=%s (%zu bits), ndim=%zu, shape=[%zu, %zu], elements=%zu, bytes=%zu\n",
           view.name,
           safetensors_dtype_name(view.dtype),
           safetensors_dtype_bitsize(view.dtype),
           view.ndim,
           view.shape[0], view.shape[1],
           view.num_elements,
           view.data_len);

    const float *loaded_weights = (const float *)view.data;
    printf("Values: [");
    for (size_t i = 0; i < view.num_elements; ++i) {
        printf("%s%.1f", (i == 0 ? "" : ", "), loaded_weights[i]);
    }
    printf("]\n");

    /* 6. Clean up handle */
    safetensors_free(handle);
    return EXIT_SUCCESS;
}
