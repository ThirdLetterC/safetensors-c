CARGO ?= cargo
CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(OS),Windows_NT)
    LDFLAGS_SYS := -lws2_32 -luserenv -lbcrypt -lntdll -ladvapi32
    EXE_EXT := .exe
else ifeq ($(UNAME_S),Darwin)
    LDFLAGS_SYS := -lpthread -lm
    EXE_EXT :=
else
    LDFLAGS_SYS := -lpthread -ldl -lm
    EXE_EXT :=
endif

.PHONY: all dist example clean

all: dist

dist:
	$(CARGO) build --release
	@mkdir -p dist/include dist/lib
	cp include/safetensors.h dist/include/safetensors.h
	@if [ -f target/release/libsafetensors_c.a ]; then \
		cp target/release/libsafetensors_c.a dist/lib/libsafetensors_c.a; \
	elif [ -f target/release/safetensors_c.lib ]; then \
		cp target/release/safetensors_c.lib dist/lib/safetensors_c.lib; \
		cp target/release/safetensors_c.lib dist/lib/libsafetensors_c.a; \
	fi

example: dist
	$(CC) $(CFLAGS) -I./dist/include examples/example.c ./dist/lib/libsafetensors_c.a $(LDFLAGS_SYS) -o dist/safetensors_example$(EXE_EXT)

clean:
	$(CARGO) clean
	rm -rf dist
