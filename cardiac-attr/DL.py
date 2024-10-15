def data_loader(list_input, list_output, 
                cache_rate = 1, augmentation = False, transform = "default",
                batch_size = 16, pixdim = (1,1,1), patch_size = (256,256,1),
                orientation = "RAS", input_interpolation = "bilinear", 
                output_interpolation = "bilinear", samrt_num_workers = 0, 
                num_samples = 30, positive = 1, negative =1 , input_lower_intensity = -70, 
                input_upper_intensity = 170, output_lower_intensity = 0,
                output_upper_intensity = 1,
                input_b_min = 0, input_b_max = 1, 
                output_b_min = 0, output_b_max = 1,
                randcroppad_tresh = 0, ImageCropForeGroundMargin = 0, 
                SegmentCropForeGroundMargin = (20,20,40),
                clip = True, task = "segmentation", data_split = "train", 
                data_loader_type = "Thread", cachedata = True,
                data_loader_num_threads = 0,
                scale_image_to_self_max = False,
                num_segment_classes = 2,
                segmentation_threshold=0.5,
                ):
    import monai
    import numpy as np
    from termcolor import cprint
    ImageCropForeGroundMargin = (ImageCropForeGroundMargin,) * len(pixdim)
    data_dictionary = [{"input_image": input_image, "output_image": output_image} 
                        for input_image, output_image  in 
                        zip(list_input,list_output)]
    if transform == "default":
        if data_split == "train":
            pass
        elif data_split == "test":
            if task == "regression":
                pass
            elif task == "segmentation":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image"], pixdim = pixdim,
                                                  mode = [input_interpolation]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image"], axcodes=orientation),
                        # monai.transforms.CropForegroundd(keys=["input_image"], source_key="input_image", margin = ImageCropForeGroundMargin),
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity, a_max = input_upper_intensity,
                                                              clip=clip, b_min = input_b_min, b_max = input_b_max),                            
                    ]
                )
                post_transform = monai.transforms.Compose(
                    [
                        monai.transforms.Invertd(
                            keys="predict_image",
                            transform=initial_transform,
                            orig_keys="input_image",
                            meta_keys="predict_image_meta_dict",
                            orig_meta_keys="input_image_meta_dict",
                            meta_key_postfix="meta_dict",
                            nearest_interp=False,
                            to_tensor=True,
                            device="cpu",
                            allow_missing_keys = True,
                        ),
                        monai.transforms.Activationsd(keys="predict_image", sigmoid=True, squared_pred = False),
                        monai.transforms.AsDiscreted(keys="predict_image", to_one_hot = num_segment_classes, threshold = segmentation_threshold),
                    ]
                )
        elif data_split == "validation":
            pass
    else: # not default transform
        initial_transform = transform
        if task == "regression" and data_split == "test":
            post_transform = monai.transforms.Compose(
                [
                    monai.transforms.Invertd(
                        keys="predict_image",
                        transform=initial_transform,
                        orig_keys="input_image",
                        meta_keys="predict_image_meta_dict",
                        orig_meta_keys="input_image_meta_dict",
                        meta_key_postfix="meta_dict",
                        nearest_interp=False,
                        to_tensor=True,
                        device="cpu",
                    ),
                ]
            )
        elif task == "segmentation" and data_split == "test":
            post_transform = monai.transforms.Compose(
                [
                    monai.transforms.Invertd(
                        keys="predict_image",
                        transform=initial_transform,
                        orig_keys="input_image",
                        meta_keys="predict_image_meta_dict",
                        orig_meta_keys="input_image_meta_dict",
                        meta_key_postfix="meta_dict",
                        nearest_interp=False,
                        to_tensor=True,
                        device="cpu",
                    ),
                    monai.transforms.Activationsd(keys="predict_image", sigmoid=True, squared_pred = True),
                    monai.transforms.AsDiscreted(keys="predict_image", to_one_hot = num_segment_classes, threshold = segmentation_threshold),
                ]
            )
        elif task == "regression" and data_split == "validation" or "train":
            post_transform = "none"
        elif task == "segmentation" and data_split == "validation" or "train":
            post_transform = monai.transforms.Compose(
                [
                    monai.transforms.AsDiscreted(keys="predict_image", argmax=True, to_onehot=num_segment_classes, threshold = segmentation_threshold),
                    monai.transforms.AsDiscreted(keys="output_image", to_onehot=num_segment_classes, threshold = segmentation_threshold),
                ]
            )
            
    if cachedata:
        # initial_dataset = monai.data.SmartCacheDataset(
        #     data = data_dictionary,
        #     transform = initial_transform,
        #     cache_rate = cache_rate,
        #     num_init_workers = samrt_num_workers,
        #     num_replace_workers = samrt_num_workers,
        #     copy_cache=True, shuffle = True,
        # )
        initial_dataset = monai.data.CacheDataset(
            data = data_dictionary,
            transform = initial_transform,
            cache_rate = cache_rate,
            num_workers = samrt_num_workers,
            copy_cache=True, 
        )
    else:
        initial_dataset = monai.data.Dataset(
            data = data_dictionary,
            transform = initial_transform,
        )
        
    if data_loader_type == "Thread":
        data_loader = monai.data.ThreadDataLoader(
            initial_dataset,
            batch_size = batch_size,
            num_workers = data_loader_num_threads,
            pin_memory = True,
            shuffle = True,
            use_thread_workers = bool(data_loader_num_threads),
            persistent_workers = bool(data_loader_num_threads),
        )
    else:
        data_loader = monai.data.DataLoader(
            initial_dataset,
            batch_size = batch_size,
            num_workers = data_loader_num_threads,
            pin_memory = True,
            shuffle = True,
            persistent_workers = bool(data_loader_num_threads),
        )
    
    check_data = monai.utils.misc.first(data_loader)
    input_shape = check_data["input_image"].shape
    output_shape = check_data["output_image"].shape  
    input_sample = np.squeeze(check_data["input_image"].numpy())
    output_sample = np.squeeze(check_data["output_image"].numpy())
    
    return_dictionary = {"data_loader" : data_loader, "initial_transform":initial_transform,
                         "input_sample":input_sample,"input_shape":input_shape, "output_sample":output_sample,
                         "output_shape":output_shape, "post_transform":post_transform}
    return return_dictionary
def model_inference_segmentation(model_url,
                                 list_images = "from-model",
                                 predict_directory = "same", 
                                 save_results = True,
                                 device = "cuda",
                                 prefix = "", 
                                 suffix = "" , 
                                 fill_holes = False, 
                                 organ_name = "from-model", 
                                 keep_largest_segment = False,
                                 num_largest_segment = 1,
                                 include_image_name = True, 
                                 cache_rate = 1,
                                 samrt_num_workers = 4,
                                 data_loader_num_threads = True,
                                 save_probabilties = False,
                                 Remove_first_dim = True,
                                 ):
    import yazdan
    import monai
    import torch
    from tqdm import tqdm
    import os
    import nibabel as nib
    from termcolor import cprint
    import SimpleITK as sitk
    import time
    from termcolor import cprint
    model_dictionary = torch.load(model_url)
    model = model_dictionary["model"]
    validation_transforms = model_dictionary["validation_transform"]
    train_transform = model_dictionary["train_transform"]
    sliding_windows_overlap = model_dictionary["sliding_windows_overlap"]
    segmented_organ = model_dictionary["segmented_organ"]
    post_transforms_test = model_dictionary["post_transforms_test"]
    test_transform = model_dictionary["test_transform"]
    single_input_shape = model_dictionary["single_input_shape"]
    sliding_window_shape = model_dictionary["sliding_window_shape"]
    if organ_name == "from-model": 
        organ_name = model_dictionary["segmented_organ"]
    try:
        params_to_torch_save = model_dictionary["params_to_torch_save"]
    except:
        cprint("old model training to params_to_torch_save", "white", "on_green")
            

    device = torch.device(device)
    model.to(device)
    if list_images == "from-model":
        list_images = model_dictionary["test_data_dictionary"]
        list_images = [x["input_image"] for x in list_images]
        
    try:
        data_loader_dict_test = yazdan.DL.data_loader(list_images, list_images,
                                            transform=test_transform,
                                            data_split="test",
                                            batch_size = 1,
                                            data_loader_type="Thread",
                                            cache_rate = cache_rate, samrt_num_workers = samrt_num_workers, 
                                            )
        data_loader_test = data_loader_dict_test["data_loader"]
    except:
        data_loader_dict_test = yazdan.DL.data_loader(list_images, list_images,
                                            pixdim = params_to_torch_save["pixdim"], transform="default", orientation = params_to_torch_save["orientation"],
                                            input_lower_intensity = params_to_torch_save["input_lower_intensity"], input_upper_intensity= params_to_torch_save["input_upper_intensity"],
                                            output_lower_intensity=params_to_torch_save["output_lower_intensity"], output_upper_intensity=params_to_torch_save["output_upper_intensity"],
                                            output_b_min=params_to_torch_save["output_b_min"], output_b_max=params_to_torch_save["output_b_max"],
                                            input_b_min=params_to_torch_save["input_b_min"], input_b_max=params_to_torch_save["input_b_max"], 
                                            task = "regression", data_split="test",
                                            clip = params_to_torch_save["clip"], batch_size = 1,
                                            data_loader_type = params_to_torch_save["data_loader_type"],  cachedata = True,
                                            cache_rate = cache_rate, samrt_num_workers = samrt_num_workers, 
                                            data_loader_num_threads = data_loader_num_threads,
                                            )
        data_loader_test = data_loader_dict_test["data_loader"]
        post_transforms_test = data_loader_dict_test["post_transform"]
        
    post_transforms_list = list(data_loader_dict_test["post_transform"].transforms)
    del post_transforms_list[-1]
    post_transforms_test_probability = monai.transforms.Compose(post_transforms_list)
            
    model.eval()
    list_segmented_urls = []
    for test_batch_data in tqdm(data_loader_test, desc = "external-inference", colour = "cyan"):
        with torch.no_grad():
            
          
            if len(single_input_shape) == 2 and len([ x for x in test_batch_data["input_image"].shape if x>1]) < 3:
                # the input images are 2D  --- this is 2D training
                image_one_dimension = [index for index, value in enumerate(list(test_batch_data["input_image"].shape)) if value == 1]
                this_is_2D_training_on_2D_images = True
            else:
                # this is either 3D images training or 2D slices of a 3D images
                image_one_dimension = [-1,]
                this_is_2D_training_on_2D_images = False
                            
            input_image = torch.squeeze(test_batch_data["input_image"], dim =  image_one_dimension[-1]).to(device)
                    
            if len(single_input_shape) == 2 and len([ x for x in input_image.shape if x>1]) < 3:
                this_is_2D_training_on_2D_images = True
            else:
                this_is_2D_training_on_2D_images = False
            if len(single_input_shape) == 2 and len([ x for x in input_image.shape if x>1]) >2:
                axial_inferer = monai.inferers.SliceInferer(roi_size= sliding_window_shape,
                                                            overlap =sliding_windows_overlap, 
                                                            sw_batch_size=1,
                                                            # cval=-1, progress=False,
                                                            )
                predict_image = axial_inferer(input_image, model)
            else:
                predict_image = monai.inferers.sliding_window_inference(input_image,
                                                                        sliding_window_shape, 1,
                                                                        model, overlap = sliding_windows_overlap, progress=False)
              
            if this_is_2D_training_on_2D_images:
                 predict_image = torch.unsqueeze(predict_image, dim = image_one_dimension[-1])

            test_batch_data["predict_image"] = predict_image
            test_batch_data_prepared = [post_transforms_test(i) for i in monai.data.decollate_batch(test_batch_data)]
            test_predicted_image, test_output_image = monai.handlers.utils.from_engine(["predict_image", "output_image"])(test_batch_data_prepared)
            
            test_batch_data_prepared_probability = [post_transforms_test_probability(i) for i in monai.data.decollate_batch(test_batch_data)]
            test_predicted_image_probability, _ = monai.handlers.utils.from_engine(["predict_image", "output_image"])(test_batch_data_prepared_probability)
            
            
            
            if this_is_2D_training_on_2D_images:
                predict_image_to_write = torch.squeeze(test_predicted_image[0], dim = 0)
                predicted_prob_to_write =  torch.squeeze(test_predicted_image_probability[0], dim = 0)
            else:
                predict_image_to_write = torch.squeeze(test_predicted_image[0])
                predicted_prob_to_write = torch.squeeze(test_predicted_image_probability[0])
            if Remove_first_dim:
                predict_image_to_write = predict_image_to_write[1,:,:,:]
                predicted_prob_to_write = predicted_prob_to_write[1,:,:,:]
                
            if fill_holes:
                hole_fill_transform = monai.transforms.FillHoles()
                predict_image_to_write = hole_fill_transform(predict_image_to_write)
           
            # predict_image_to_write = predict_image_to_write.bool()
            predict_image_to_write = predict_image_to_write.to(torch.uint8)
            test_image_url = test_batch_data["input_image_meta_dict"]["filename_or_obj"][0]
            test_image_name = os.path.basename(test_image_url).replace(".nii.gz", "")
            header = nib.load(test_image_url).header
            affine = nib.load(test_image_url).affine
            predicted_nifti = nib.Nifti1Image(predict_image_to_write.detach().cpu().numpy(), affine = affine,header=header)
            predicted_probability_nifti = nib.Nifti1Image(predicted_prob_to_write.detach().cpu().numpy(), affine = affine,header=header)
            if include_image_name:
                predict_segment_name = f"{prefix}{test_image_name}{suffix}-{organ_name}.nii.gz"
            else:
               predict_segment_name = f"{prefix}{organ_name}-segmentations{suffix}.nii.gz" 
            
            if save_results:
                if predict_directory == "same":
                    output_segment_url = os.path.join(os.path.dirname(test_image_url), predict_segment_name)
                    nib.save(predicted_nifti,os.path.join(os.path.dirname(test_image_url), predict_segment_name))
                    nib.save(predicted_probability_nifti,os.path.join(os.path.dirname(test_image_url), predict_segment_name.replace(".nii.gz", "--probability.nii.gz")))
                else:
                    output_segment_url = os.path.join(predict_directory, predict_segment_name)
                    nib.save(predicted_nifti,os.path.join(predict_directory, predict_segment_name))
                    nib.save(predicted_probability_nifti,os.path.join(predict_directory, predict_segment_name.replace(".nii.gz", "--probability.nii.gz")))
                list_segmented_urls.append(output_segment_url)
            
    if keep_largest_segment:
        for network_output_url in tqdm(list_segmented_urls, desc = f"keeping {num_largest_segment} largest segments", colour = "red"):
            try:
                segment_large_objects = yazdan.image.keep_largest_segments(network_output_url, numbner_of_objects = num_largest_segment)
                sitk.WriteImage(segment_large_objects, network_output_url)
            except:
                cprint(f"keep largest segment error on url: {network_output_url}", "white", "on_cyan")
        
        
    inference_results = {}
    output_url = os.path.join(predict_directory, predict_segment_name)
    inference_results["predict_image_to_write"] = predict_image_to_write 
    inference_results["test_batch_data"] = test_batch_data
    inference_results["test_predicted_image"] = test_predicted_image[0]
    inference_results["this_is_2D_training_on_2D_images"] = this_is_2D_training_on_2D_images
    inference_results["output_url"] = output_url
    return inference_results
