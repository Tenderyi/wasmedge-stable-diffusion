use wasmedge_stable_diffusion::stable_diffusion_interface::{ImageType, SdTypeT, RngTypeT, SampleMethodT, ScheduleT};
use wasmedge_stable_diffusion::{BaseFunction, Context, Quantization, StableDiffusion, Task};
use clap::{crate_version, Arg, ArgAction, Command};

use std::str::FromStr;

use rand::Rng;
use std::time::{SystemTime, UNIX_EPOCH};

//Sampling Methods
const SAMPLE_METHODS: [&str; 8] = [
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
];

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const SCHEDULE_STR: [&str; 4] = [
    "default",
    "discrete",
    "karras",
    "ays",
];

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
            Arg::new("taesd_path")
                .long("taesd")
                .value_name("TAESD_PATH")
                .help("path to taesd. Using Tiny AutoEncoder for fast decoding (low quality)")
                .default_value(""),
        )

        .arg(
            Arg::new("embeddings_path")
                .long("embd-dir")
                .value_name("EMBEDDING_PATH")
                .help("path to embeddings.")
                .default_value(""),
        )
        .arg(
            Arg::new("stacked_id_embd_dir")
                .long("stacked-id-embd-dir")
                .value_name("STACKED_ID_EMBD_DIR")
                .help("path to PHOTOMAKER stacked id embeddings.")
                .default_value(""),
        )
        .arg(
            Arg::new("input_id_images_dir")
                .long("input-id-images-dir")
                .value_name("INPUT_ID_IMAGES_DIR")
                .help("path to PHOTOMAKER input id images dir.")
                .default_value(""),
        )
        .arg(
            Arg::new("normalize_input")
                .long("normalize-input")
                .help("normalize PHOTOMAKER input id images")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("lora_model_dir")
                .long("lora-model-dir")
                .value_name("LORA_MODEL_DIR")
                .help("lora model directory")
                .default_value(""),
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
        .arg(
            Arg::new("negative_prompt")
                .short('n')
                .long("negative-prompt")
                .value_name("NEGATIVE_PROMPT")
                .help("the negative prompt")
                .default_value(""),
        )
        .arg(
            Arg::new("cfg_scale")
                .long("cfg-scale")
                .value_parser(clap::value_parser!(f32))
                .value_name("CFG_SCALE")
                .help("unconditional guidance scale: (default: 7.0)")
                .default_value("7.0"),
        )
        .arg(
            Arg::new("strength")
                .long("strength")
                .value_parser(clap::value_parser!(f32))
                .value_name("STRENGTH")
                .help("strength for noising/unnoising")
                .default_value("0.75"),
        )
        .arg(
            Arg::new("style_ratio")
                .long("style-ratio")
                .value_parser(clap::value_parser!(f32))
                .value_name("STYLE_RATIO")
                .help("strength for keeping input identity")
                .default_value("20.0"),
        )
        .arg(
            Arg::new("control_strength")
                .long("control-strength")
                .value_parser(clap::value_parser!(f32))
                .value_name("CONTROL-STRENGTH")
                .help("strength to apply Control Net (default: 0.9) 1.0 corresponds to full destruction of information in init image")
                .default_value("0.9"),
        )
        .arg(
            Arg::new("sampling_method")
                .long("sampling-method")
                .value_parser([
                    "euler_a",
                    "euler",
                    "heun",
                    "dpm2",
                    "dpm++2s_a",
                    "dpm++2m",
                    "dpm++2mv2",
                    "lcm",
                ])
                .value_name("SAMPLING_METHOD")
                .help("the sampling method, include values {euler, euler_a, heun, dpm2, dpm++2s_a, dpm++2m, dpm++2mv2, lcm}")
                .default_value("euler_a"),
        )
        .arg(
            Arg::new("steps")
                .long("steps")
                .value_parser(clap::value_parser!(i32))
                .value_name("STEPS")
                .help("number of sample steps")
                .default_value("20"),
        )           
        .arg(
            Arg::new("rng")
                .long("rng")
                .value_name("RNG")
                .value_parser([
                    "std_default",
                    "cuda",
                ])
                .help("RNG (default: std_default)")
                .default_value("std_default"),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .short('s')
                .value_parser(clap::value_parser!(i32))
                .value_name("SEED")
                .help("RNG seed (default: 42, use random seed for < 0)")
                .default_value("42"),
        )
        .arg(
            Arg::new("batch_count")
                .long("batch-count")
                .short('b')
                .value_parser(clap::value_parser!(i32))
                .value_name("BATCH_COUNT")
                .help("number of images to generate.")
                .default_value("1"),
        )
        .arg(
            Arg::new("schedule")
                .long("schedule")
                .value_name("SCHEDULE")
                .value_parser([
                    "default",
                    "discrete",
                    "karras",
                    "ays",
                ])
                .help("Denoiser sigma schedule")
                .default_value("default"),
        )
        .arg(
            Arg::new("clip_skip")
                .long("clip-skip")
                .value_parser(clap::value_parser!(i32))
                .value_name("CLIP_SKIP")
                .help("ignore last layers of CLIP network; 1 ignores none, 2 ignores one layer (default: -1), <= 0 represents unspecified, will be 1 for SD1.x, 2 for SD2.x")
                .default_value("-1"),
        )                        
        .arg(
            Arg::new("vae_tiling")
                .long("vae-tiling")
                .help("process vae in tiles to reduce memory usage")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("clip_on_cpu")
                .long("clip-on-cpu")
                .help("clip on cpu")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("control_net_cpu")
                .long("control-net-cpu")
                .help("keep controlnet in cpu (for low vram)")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("vae_on_cpu")
                .long("vae-on-cpu")
                .help("vae on cpu")
                .action(ArgAction::SetTrue),
        )
        .after_help("run at the dir of .wasm, Example:wasmedge --dir .:. ./target/wasm32-wasi/release/wasmedge_stable_diffusion_example.wasm -m ../../models/stable-diffusion-v1-4-Q8_0.gguf -M img2img\n")
        .get_matches();
    
    
    //init the paraments--------------------------------------------------------------
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
    
    //taesd_path
    let taesd_path = matches.get_one::<String>("taesd_path")
        .ok_or(String::from(
            "Fail to parse the `taesd_path` option from the command line.",
        ))?;
    println!("[INFO] taesd_path: {}", taesd_path);

    //embeddings_path
    let embeddings_path = matches.get_one::<String>("embeddings_path")
        .ok_or(String::from(
            "Fail to parse the `embeddings_path` option from the command line.",
        ))?;
    println!("[INFO] embeddings_path: {}", embeddings_path);

    //stacked_id_embd_dir
    let stacked_id_embd_dir = matches
        .get_one::<String>("stacked_id_embd_dir")
        .ok_or(String::from(
            "Fail to parse the `stacked_id_embd_dir` option from the command line.",
        ))?;
    println!("[INFO] stacked_id_embd_dir: {}", stacked_id_embd_dir);

    //input_id_images_dir
    let input_id_images_dir = matches
        .get_one::<String>("input_id_images_dir")
        .ok_or(String::from(
            "Fail to parse the `input_id_images_dir` option from the command line.",
        ))?;
    println!("[INFO] input_id_images_dir: {}", input_id_images_dir);

    //normalize-input
    let normalize_input = matches.get_flag("normalize_input");
    println!("[INFO] normalize_input: {}", normalize_input);

    //lora_model_dir
    let lora_model_dir = matches
        .get_one::<String>("lora_model_dir")
        .ok_or(String::from(
            "Fail to parse the `lora_model_dir` option from the command line.",
        ))?;
    println!("[INFO] lora_model_dir: {}", lora_model_dir);

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

    //negative_prompt
    let negative_prompt = matches
        .get_one::<String>("negative_prompt")
        .ok_or(String::from(
            "Fail to parse the `negative-prompt` option from the command line.",
        ))?;
    println!("[INFO] negative_prompt: {}", negative_prompt);

    //cfg_scale
    let cfg_scale = matches.get_one::<f32>("cfg_scale").unwrap();
    println!("[INFO] cfg_scale: {}", *cfg_scale);

    //strength
    let strength = matches.get_one::<f32>("strength").unwrap();
    if *strength > 1.0 {
        return Err("Error: can only work with strength in [0.0, 1.0]".into());
    }
    println!("[INFO] strength: {}", *strength);

    //style_ratio
    let style_ratio = matches.get_one::<f32>("style_ratio").unwrap();
    if *style_ratio > 100.0 {
        return Err("Error: can only work with style_ratio in [0.0, 100.0]".into());
    }
    println!("[INFO] style_ratio: {}", *style_ratio);

    //control_strength
    let control_strength = matches.get_one::<f32>("control_strength").unwrap();
    if *control_strength > 1.0 {
        return Err("Error: can only work with control_strength in [0.0, 1.0]".into());
    }
    println!("[INFO] control_strength: {}", *control_strength);
    
    //sampling_method
    let sampling_method_selected = matches
        .get_one::<String>("sampling_method")
        .ok_or(String::from(
            "Fail to parse the `sampling_method` option from the command line.",
        ))?;
    // Find the index of sampling_method_selected in SAMPLE_METHODS
    let sample_method_found = SAMPLE_METHODS
        .iter()
        .position(|&method| method == sampling_method_selected)
        .ok_or(format!("Invalid sampling method: {}",sampling_method_selected))?;
    // Convert an index to an enumeration value
    let sample_method = SampleMethodT::from_index(sample_method_found)?;
    println!("[INFO] sampling_method: {}, enum: {:?}", sampling_method_selected, sample_method);

    //sample_steps
    let sample_steps = matches.get_one::<i32>("steps").unwrap();
    if *sample_steps < 0 {
        return Err("Error: the sample_steps must be greater than 0".into());
    }
    println!("[INFO] sample_steps: {}", *sample_steps);

    //rng_type
    let mut rng_type = RngTypeT::StdDefaultRng;
    let rng_type_str = matches.get_one::<String>("rng").unwrap();
    if rng_type_str == "cuda"{
        rng_type =  RngTypeT::CUDARng;
    }
    println!("[INFO] rng: {}", rng_type_str);

    //seed
    let seed_str = matches.get_one::<i32>("seed").unwrap();
    let mut seed  = *seed_str;
    // let mut seed: i32 = seed_str.parse().expect("Failed to parse seed as i32");
    if seed < 0 {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        let current_time_secs = current_time.as_secs() as u32;

        let mut rng = rand::thread_rng();
        seed = ((rng.gen::<u32>() ^ current_time_secs) & i32::MAX as u32) as i32; // 将结果限制在 i32 范围内
    }
    println!("[INFO] seed: {}", seed);

    //batch_count
    let batch_count = matches.get_one::<i32>("batch_count").unwrap();
    println!("[INFO] batch_count: {}", *batch_count);

    //schedule
    let schedule_selected = matches.get_one::<String>("schedule").unwrap();
    let schedule_found = SCHEDULE_STR
        .iter()
        .position(|&method| method == schedule_selected)
        .ok_or(format!("Invalid sampling method: {}",schedule_selected))?;
    // Convert an index to an enumeration value
    let schedule = ScheduleT::from_index(schedule_found)?;
    println!("[INFO] sampling_method: {}, enum: {:?}", schedule_selected, schedule);

    
    //clip_skip
    let clip_skip = matches.get_one::<i32>("clip_skip").unwrap();
    println!("[INFO] clip_skip: {}", *clip_skip);

    //vae_tiling
    let vae_tiling = matches.get_flag("vae_tiling");
    println!("[INFO] vae_tiling: {}", vae_tiling);

    //clip_on_cpu
    let clip_on_cpu = matches.get_flag("clip_on_cpu");
    println!("[INFO] clip_on_cpu: {}", clip_on_cpu);

    //control_net_cpu
    let control_net_cpu = matches.get_flag("control_net_cpu");
    println!("[INFO] control_net_cpu: {}", control_net_cpu);

    //vae_on_cpu
    let vae_on_cpu = matches.get_flag("vae_on_cpu");
    println!("[INFO] vae_on_cpu: {}", vae_on_cpu);



    //-----------------------------------------------------------------------
    //run the model
    let context = StableDiffusion::new(task, sd_model, 
        taesd_path, lora_model_dir, embeddings_path, stacked_id_embd_dir,
        vae_tiling, 

        rng_type,
        schedule,
        clip_on_cpu,
        control_net_cpu,
        vae_on_cpu
    );
    match sd_mode.as_str(){
        "txt2img" => {
            println!("txt2img");
            if let Context::TextToImage(mut text_to_image) = context.create_context().unwrap() {
                text_to_image
                    .set_input_id_images_dir(input_id_images_dir)
                    .set_normalize_input(normalize_input)
                    .set_prompt(prompt)
                    .set_negative_prompt(negative_prompt)
                    .set_cfg_scale(*cfg_scale)
                    .set_style_ratio(*style_ratio)
                    .set_control_strength(*control_strength)
                    .set_sample_method(sample_method)
                    .set_sample_steps(*sample_steps)
                    .set_seed(seed)
                    .set_batch_count(*batch_count)
                    .set_clip_skip(*clip_skip)

                    .set_output_path(output)
                    .generate()
                    .unwrap();
            }
        },
        "img2img" => {
            println!("img2img");
            if let Context::ImageToImage(mut image_to_image) = context.create_context().unwrap() {
                image_to_image
                    .set_input_id_images_dir(input_id_images_dir)
                    .set_normalize_input(normalize_input)
                    .set_prompt(prompt)
                    .set_negative_prompt(negative_prompt)
                    .set_cfg_scale(*cfg_scale)
                    .set_style_ratio(*style_ratio)
                    .set_control_strength(*control_strength)
                    .set_sample_method(sample_method)
                    .set_sample_steps(*sample_steps)
                    .set_seed(seed)
                    .set_batch_count(*batch_count)
                    .set_clip_skip(*clip_skip)

                    .set_image(ImageType::Path(init_img.expect("no init img")))
                    .set_strength(*strength)
                    .set_output_path(output)
                    .generate()
                    .unwrap();
            }
        },
        "convert" => {
            println!("Error: convert!");
        },
        _ => {
            println!("Error: this mode isn't supported!");
        }
    }

    return Ok(());
}

// #[derive(Debug, Default, Deserialize, Serialize)]
// struct Options {
//     #[serde(rename = "mode")]
//     mode: String,
//     #[serde(rename = "model")]
//     model: String,
//     #[serde(rename = "init-img")]
//     init_img: String,
//     #[serde(rename = "output")]
//     output: String,
//     #[serde(rename = "prompt")]
//     prompt: String,

//     //edit options
//     #[serde(skip_serializing_if = "Option::is_none", rename = "negative-prompt")]
//     negative_prompt: Option<String>,
//     #[serde(skip_serializing_if = "Option::is_none", rename = "sampling-method")]
//     sampling_method: Option<String>,
//     #[serde(skip_serializing_if = "Option::is_none")]
//     steps: Option<i32>,
// }


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