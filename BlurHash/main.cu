#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <string.h>
#include <cstdint>
#include <math.h>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const char* blurHashForFile(const char* filename);
const char* blurHashForPixels(int width, int height, uint8_t* rgb, size_t bytesPerRow);
static char* encode_int(int value, int length, char* destination);
static int encodeDC(float r, float g, float b);
static int encodeAC(float r, float g, float b, float maximumValue);
void cleanUp(float* allFactor, float*** factors, float* d_allFactor, uint8_t* d_rgb);

// xComponents and yComponents are in <1,9>
const static int xComponents = 8; 
const static int yComponents = 8;

// Utilities same as in CPU
static inline int linearTosRGB(float value) {
	float v = fmaxf(0, fminf(1, value));
	if (v <= 0.0031308) return v * 12.92 * 255 + 0.5;
	else return (1.055 * powf(v, 1 / 2.4) - 0.055) * 255 + 0.5;
}

__device__ static inline float sRGBToLinear(int value) {
	float v = (float)value / 255;
	if (v <= 0.04045) return v / 12.92;
	else return powf((v + 0.055) / 1.055, 2.4);
}

static inline float signPow(float value, float exp) {
	return copysignf(powf(fabsf(value), exp), value);
}

// Main
int main(int argc, char** argv)
{
	if (argc != 2) {
		fprintf(stderr, "Usage: %s imagefile\n", argv[0]);
		return 1;
	}

	auto start = std::chrono::high_resolution_clock::now();

	const char* hash = blurHashForFile(argv[1]);
	if (!hash) {
		fprintf(stderr, "Failed to load image file \"%s\".\n", argv[3]);
		return 1;
	}

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	printf("%s\n", hash);
	printf("\n[App] Total: %d ms\n", duration.count());

	return 0;
}

__global__ void getHash(int yComponents, int xComponents, int width, int height, uint8_t* d_rgb, size_t bytesPerRow, float* d_factors)
{
	/*
	* <summary>
	*	multiplyBasisFunction from CPU but for one pixel and all of components
	* </summary>
	*/
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= width * height) return;

	int x = id % width;
	int y = id / width;

	for (int i = 0; i < yComponents; i++)
	{
		for (int j = 0; j < xComponents; j++)
		{
			float basis = cosf(M_PI * j * x / width) * cosf(M_PI * i * y / height);
			float normalisation = (j == 0 && i == 0) ? 1.0 : 2.0;
			float scale = normalisation / (width * height);

			atomicAdd(d_factors + i * xComponents * 3 + j * 3, scale * basis * sRGBToLinear(d_rgb[3 * x + 0 + y * bytesPerRow]));
			atomicAdd(d_factors + i * xComponents * 3 + j * 3 + 1, scale * basis * sRGBToLinear(d_rgb[3 * x + 1 + y * bytesPerRow]));
			atomicAdd(d_factors + i * xComponents * 3 + j * 3 + 2, scale * basis * sRGBToLinear(d_rgb[3 * x + 2 + y * bytesPerRow]));
		}
	}
}

const char* blurHashForFile(const char* filename) {
	int width, height, channels;
	
	auto start = std::chrono::high_resolution_clock::now();

	unsigned char* data = stbi_load(filename, &width, &height, &channels, 3);
	
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf("[App] Reading: %d ms\n", duration.count());
	
	if (!data) return NULL;

	const char* hash = blurHashForPixels(width, height, data, width * 3);

	stbi_image_free(data);

	return hash;
}

const char* blurHashForPixels(int width, int height, uint8_t* rgb, size_t bytesPerRow) {

	uint8_t* d_rgb;

	int threads, blocks;
	threads = 2 << 9;
	blocks = ceil((double)width * height / threads);

	auto start = std::chrono::high_resolution_clock::now();

	static char buffer[2 + 4 + (9 * 9 - 1) * 2 + 1];

	// allFactor same as factors in CPU but converted into one 1D array for simplicyty of CUDA
	float* allFactor = (float*) malloc (yComponents * xComponents * 3 * sizeof(float));
	float*** factors = (float***) malloc (yComponents * sizeof(float**));
	float* d_allFactor;

	if (!allFactor || !factors) return nullptr;

	// factors initialization
	for (int i = 0; i < yComponents; i++)
	{
		*(factors + i) = (float**) malloc (xComponents * sizeof(float*));
		if (!(factors + i))
			return nullptr;

		for (int j = 0; j < xComponents; j++)
		{
			factors[i][j] = allFactor + (i * xComponents * 3) + (j * 3);
		}
	}

	memset(allFactor, 0, yComponents * xComponents * 3 * sizeof(float));
	cudaMalloc(&d_allFactor, yComponents * xComponents * 3 * sizeof(float));
	cudaMemset(d_allFactor, 0, yComponents * xComponents * 3 * sizeof(float));

	cudaMalloc(&d_rgb, 3 * width * height * sizeof(uint8_t));
	cudaMemcpy(d_rgb, rgb, 3 * width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf("[App] Memory management: %d ms\n", duration.count());

	start = std::chrono::high_resolution_clock::now();

	getHash <<<blocks, threads >>> (yComponents, xComponents, width, height, d_rgb, bytesPerRow, d_allFactor);
	cudaDeviceSynchronize();

	stop = std::chrono::high_resolution_clock::now();

	cudaMemcpy(allFactor, d_allFactor, yComponents * xComponents * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	float* dc = factors[0][0];
	float* ac = dc + 3;
	int acCount = xComponents * yComponents - 1;
	char* ptr = buffer;

	int sizeFlag = (xComponents - 1) + (yComponents - 1) * 9;
	ptr = encode_int(sizeFlag, 1, ptr);

	float maximumValue;
	if (acCount > 0) {
		float actualMaximumValue = 0;
		for (int i = 0; i < acCount * 3; i++) {
			actualMaximumValue = fmaxf(fabsf(ac[i]), actualMaximumValue);
		}

		int quantisedMaximumValue = fmaxf(0, fminf(82, floorf(actualMaximumValue * 166 - 0.5)));
		maximumValue = ((float)quantisedMaximumValue + 1) / 166;
		ptr = encode_int(quantisedMaximumValue, 1, ptr);
	}
	else {
		maximumValue = 1;
		ptr = encode_int(0, 1, ptr);
	}

	ptr = encode_int(encodeDC(dc[0], dc[1], dc[2]), 4, ptr);

	for (int i = 0; i < acCount; i++) {
		ptr = encode_int(encodeAC(ac[i * 3 + 0], ac[i * 3 + 1], ac[i * 3 + 2], maximumValue), 2, ptr);
	}

	*ptr = 0;
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf("[App] Hash computing: %d ms\n", duration.count());
	cleanUp(allFactor, factors, d_allFactor, d_rgb);
	return buffer;
}

void cleanUp(float* allFactor, float*** factors, float* d_allFactor, uint8_t* d_rgb)
{
	auto start = std::chrono::high_resolution_clock::now();

	free(allFactor);
	for (int i = 0; i < yComponents; i++)
	{
		free(factors[i]);
	}
	free(factors);
	cudaFree(d_allFactor);
	cudaFree(d_rgb);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	printf("[App] Cleaning: %d ms\n", duration.count());
}

/*
* <summary>
*	Same functions as in CPU implementation
* </summary>
*/
static int encodeDC(float r, float g, float b) {
	int roundedR = linearTosRGB(r);
	int roundedG = linearTosRGB(g);
	int roundedB = linearTosRGB(b);
	return (roundedR << 16) + (roundedG << 8) + roundedB;
}

static int encodeAC(float r, float g, float b, float maximumValue) {
	int quantR = fmaxf(0, fminf(18, floorf(signPow(r / maximumValue, 0.5) * 9 + 9.5)));
	int quantG = fmaxf(0, fminf(18, floorf(signPow(g / maximumValue, 0.5) * 9 + 9.5)));
	int quantB = fmaxf(0, fminf(18, floorf(signPow(b / maximumValue, 0.5) * 9 + 9.5)));

	return quantR * 19 * 19 + quantG * 19 + quantB;
}

static char characters[84] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

static char* encode_int(int value, int length, char* destination) {
	int divisor = 1;
	for (int i = 0; i < length - 1; i++) divisor *= 83;

	for (int i = 0; i < length; i++) {
		int digit = (value / divisor) % 83;
		divisor /= 83;
		*destination++ = characters[digit];
	}
	return destination;
}
