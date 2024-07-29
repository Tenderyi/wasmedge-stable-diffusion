# wasmedge-stable-diffusion
A Rust library for using stable diffusion functions when the Wasi is being executed on WasmEdge.
## Set up WasmEdge
git clone https://github.com/WasmEdge/WasmEdge.git
```python
cd WasmEdge
cmake -GNinja -Bbuild -DCMAKE_BUILD_TYPE=Release -DWASMEDGE_BUILD_TESTS=OFF -DWASMEDGE_PLUGIN_STABLEDIFFUSION=On -DWASMEDGE_USE_LLVM=OFF
cmake --build build
sudo cmake --install build
```

## Download Model
Download the weights or quantized model from the following command.  
You also can use our example to quantize the weights by yourself.

stable-diffusion v1.4: [second-state/stable-diffusion-v-1-4-GGUF](https://huggingface.co/second-state/stable-diffusion-v-1-4-GGUF)  
stable-diffusion v1.5: [second-state/stable-diffusion-v1-5-GGUF](https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF)  
stable-diffusion v2.1: [second-state/stable-diffusion-2-1-GGUF](https://huggingface.co/second-state/stable-diffusion-2-1-GGUF)

```
curl -L -O https://huggingface.co/second-state/stable-diffusion-v-1-4-GGUF/resolve/main/sd-v1-4.ckpt
curl -L -O https://huggingface.co/second-state/stable-diffusion-v-1-4-GGUF/resolve/main/stable-diffusion-v1-4-Q8_0.gguf
```

## Run the example
The example uses the stable-diffusion-v-1-4-GGUF model, which currently only supports txt2img and img2img.
```
cargo build --target wasm32-wasi --release
wasmedge --dir .:. ./target/wasm32-wasi/release/wasmedge_stable_diffusion_example.wasm
```

Then you can see the three new files.
1. sd-v1-4-Q8_0.gguf: is the quantization of sd-v1-4
2. output.png: an image with a cat
3. output2.png: an image of a cat with blue eyes.


## parameter settings
- [ ] -h, --help                                    show this help message and exit<br>
- [x] -M, --mode [MODEL]                    run mode (txt2img or img2img or convert, default: txt2img)
- [ ] -t, --threads N                             number of threads to use during computation (default: -1).If threads <= 0, then threads will be set to the number of CPU physical cores
- [x] -m, --model [MODEL]                   path to model
- [ ] --vae [VAE]                                 path to vae
- [ ] --taesd [TAESD_PATH]                 path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)
- [ ] --control-net [CONTROL_PATH]     path to control net model
- [ ] --embd-dir [EMBEDDING_PATH]    path to embeddings.
- [ ] --stacked-id-embd-dir [DIR]          path to PHOTOMAKER stacked id embeddings.
- [ ] --input-id-images-dir [DIR]           path to PHOTOMAKER input id images dir.
- [ ] --normalize-input                         normalize PHOTOMAKER input id images
- [ ] --upscale-model [ESRGAN_PATH]   path to esrgan model. Upscale images after generate, just RealESRGAN_x4plus_anime_6B supported by now.
- [ ] --upscale-repeats                         Run the ESRGAN upscaler this many times (default 1)
- [ ] --type [TYPE]                              weight type (f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0)If not specified, the default is the type of the weight file.
- [ ] --lora-model-dir [DIR]                   lora model directory
- [x] -i, --init-img [IMAGE]                   path to the input image, required by img2img
- [ ] --control-image [IMAGE]               path to image condition, control net
- [x] -o, --output OUTPUT                    path to write result image to (default: ./output.png)
- [x] -p, --prompt [PROMPT]                 the prompt to render
- [ ] -n, --negative-prompt PROMPT      the negative prompt (default: "")
- [ ] --cfg-scale SCALE                        unconditional guidance scale: (default: 7.0)
- [ ] --strength STRENGTH                   strength for noising/unnoising (default: 0.75)
- [ ] --style-ratio STYLE-RATIO             strength for keeping input identity (default: 20%)
- [ ] --control-strength STRENGTH        strength to apply Control Net (default: 0.9) 1.0 corresponds to full destruction of information in init image
- [ ] -H, --height H                              image height, in pixel space (default: 512)
- [ ] -W, --width W                              image width, in pixel space (default: 512)
- [ ] --sampling-method {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}                                                                sampling method (default: "euler_a")
- [ ] --steps  STEPS                             number of sample steps (default: 20)
- [ ] --rng {std_default, cuda}              RNG (default: cuda)
- [ ] -s SEED, --seed SEED                   RNG seed (default: 42, use random seed for < 0)
- [ ] -b, --batch-count COUNT              number of images to generate.
- [ ] --schedule {discrete, karras, ays}  Denoiser sigma schedule (default: discrete)
- [ ] --clip-skip N                                ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1), <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x
- [ ] --vae-tiling                                  process vae in tiles to reduce memory usage
- [ ] --control-net-cpu                         keep controlnet in cpu (for low vram)
- [ ] --canny                                      apply canny preprocessor (edge detection)
- [ ] --color                                        Colors the logging tags according to level
- [ ] -v, --verbose                               print extra info


