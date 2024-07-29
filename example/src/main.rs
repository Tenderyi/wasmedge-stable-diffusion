use wasmedge_stable_diffusion::stable_diffusion_interface::{ImageType, SdTypeT};
use wasmedge_stable_diffusion::{BaseFunction, Context, Quantization, StableDiffusion, Task};
use clap::{crate_version, Arg, ArgAction, Command};

use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("wasmedge-stable-diffusion")
        .version(crate_version!())
        .arg(
            Arg::new("mode")
                .short('M')
                .long("mode")
                .value_parser([
                    "txt2img",
                    "img2img",
                    "convert",
                ])
                .value_name("MODE")
                .help("run mode (txt2img or img2img or convert)")
                .default_value("txt2img"),
        )
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("MODEL")
                .help("path to model")
                .default_value("stable-diffusion-v1-4-Q8_0.gguf"),
        )
        .arg(
            Arg::new("init_img")
                .short('i')
                .long("init-img")
                .value_name("INIT_IMG")
                .help("path to the input image, required by img2img")
                .default_value("./output.png"),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_DIR")
                .help("path to write result image")
                .default_value("./output2.png"),
        )
        .arg(
            Arg::new("prompt")
                .short('p')
                .long("prompt")
                .value_name("PROMPT")
                .help("the prompt to render")
                .default_value("a lovely cat"),
        )
        .after_help("run at the dir of .wasm, Example:wasmedge --dir .:. ./target/wasm32-wasi/release/wasmedge_stable_diffusion_example.wasm -m ../../models/stable-diffusion-v1-4-Q8_0.gguf -M img2img\n")
        .get_matches();
    
    //init the paraments
    //mode, include "txt2img","img2img",----------"convert" is not yet-------.
    let sd_mode = matches
        .get_one::<String>("mode")
        .ok_or(String::from(
            "Fail to parse the `mode` option from the command line.",
        ))?;
    let task = Task::from_str(sd_mode)?;
    println!("[INFO] mode: {}", sd_mode);
    
    //model
    let sd_model = matches
        .get_one::<String>("model")
        .ok_or(String::from(
            "Fail to parse the `model` option from the command line.",
        ))?;
    println!("[INFO] model: {}", sd_model);
    
    //init_img, used only for img2img
    let init_img = if sd_mode == "img2img" {
        let img = matches
            .get_one::<String>("init_img")
            .ok_or(String::from(
                "Fail to parse the `init-img` option from the command line.",
            ))?;
        println!("[INFO] init_img: {}", img);
        Some(img)
    } else {
        None
    };
    
    //output
    let output = matches
        .get_one::<String>("output")
        .ok_or(String::from(
            "Fail to parse the `output` option from the command line.",
        ))?;
    println!("[INFO] output: {}", output);

    //prompt
    let prompt = matches
        .get_one::<String>("prompt")
        .ok_or(String::from(
            "Fail to parse the `prompt` option from the command line.",
        ))?;
    println!("[INFO] prompt: {}", prompt);


    

    //run the model
    let context = StableDiffusion::new(task, sd_model);
    match context.create_context() {
        Ok(Context::TextToImage(mut text_to_image)) => {
            println!("[INFO]text2img");
            text_to_image
                .set_prompt(prompt)
                .set_output_path(output)
                .generate()?;
        }
        Ok(Context::ImageToImage(mut image_to_image)) => {
            println!("[INFO]ImageToImage");
            image_to_image
                .set_prompt(prompt)
                .set_image(ImageType::Path(init_img.expect("no init img")))
                .set_output_path(output)
                .generate()?;
        }
        Err(e) => return Err(e.into()),
    }

    return Ok(());
}

    // if you downloaded ckpt weights, you can use convert() to quantize the ckpt weight to gguf.
    // For running other models, you need to change the model path of the following functions. 
    // let quantization =
    //     Quantization::new("./sd-v1-4.ckpt", "stable-diffusion-v1-4-Q8_0.gguf", SdTypeT::SdTypeQ8_0);
    // quantization.convert().unwrap();

    // let context = StableDiffusion::new(Task::TextToImage, "stable-diffusion-v1-4-Q8_0.gguf");
    // if let Context::TextToImage(mut text_to_image) = context.create_context().unwrap() {
    //     text_to_image
    //         .set_prompt("a lovely cat")
    //         .set_output_path("output.png")
    //         .generate()
    //         .unwrap();
    // }

    // let context = StableDiffusion::new(Task::ImageToImage, "stable-diffusion-v1-4-Q8_0.gguf");
    // if let Context::ImageToImage(mut image_to_image) = context.create_context().unwrap() {
    //     image_to_image
    //         .set_prompt("with blue eyes")
    //         .set_image(ImageType::Path("output.png"))
    //         .set_output_path("output2.png")
    //         .generate()
    //         .unwrap();
    // }