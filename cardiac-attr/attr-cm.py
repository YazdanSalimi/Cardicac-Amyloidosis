def prepare_tb_to_rib_seg(tb_url,
                          template_image_url = r"H:\ATTR\AI-Tasks\Task0_WBPlanarSegmentation\dataset-prepared-registration\spect_ASC_2D\SPECTSEG_NM870_0000.nii.gz",
                          target_url = "none",
                          max_percentile = 99,
                          ):
    import SimpleITK as sitk
    import numpy as np
    import yazdan
    tempalate_image = sitk.ReadImage(template_image_url)
    image = sitk.ReadImage(tb_url)
    trreshold, crop_1, crop_2 = yazdan.image.CropBckg(image, treshold= (np.percentile(sitk.GetArrayFromImage(image), 90), np.percentile(sitk.GetArrayFromImage(image), 99)))
    # yazdan.image.view([crop_1])
    
    
    image_norm, image_norm_no_clip = yazdan.image.sitk_rescale(image, input_max=np.percentile(sitk.GetArrayFromImage(image), max_percentile))

    array_expanded_noclip = np.expand_dims(np.transpose(sitk.GetArrayFromImage(image_norm_no_clip), (1,0)), axis=1)
    image_updated_noclip = sitk.GetImageFromArray(array_expanded_noclip)
    image_updated_noclip.SetSpacing((image_norm_no_clip.GetSpacing()[0], ) * 3)
    image_updated_noclip = sitk.DICOMOrient(image_updated_noclip, "SLA")
    image_updated_noclip.SetDirection(tempalate_image.GetDirection())
    
    array_expanded = np.expand_dims(np.transpose(sitk.GetArrayFromImage(image_norm), (1,0)), axis=1)
    image_updated = sitk.GetImageFromArray(array_expanded)
    image_updated.SetSpacing((image_norm.GetSpacing()[0], ) * 3)
    image_updated = sitk.DICOMOrient(image_updated, "SLA")
    image_updated.SetDirection(tempalate_image.GetDirection())
    if target_url != "none":
        sitk.WriteImage(image_updated, target_url)
        sitk.WriteImage(image_updated_noclip, target_url.replace(".nii.gz", "_noClip.nii.gz"))
    return image_updated

def TB_rib_segment(lsit_tb_urls,
                   model_directory = r"C:\YazdanFiles\ATTR\AI-Tasks\Task0_WBPlanarSegmentation\dataset-prepared-registration\model_directory\swin2d-large",
                   predict_directory = "same",
                   prefix = "",
                   suffix = "",
                   Remove_first_dim = True,
                   model_index = 0,
                   max_percentile = 99,
                   keep_largest_segment = False,
                   num_largest_segment = 2,
                   organ_name = "from-model",
                   ):
    import yazdan
    from glob import glob
    from natsort import os_sorted
    import os
    list_tb_updated_urls = []
    for tb_url in lsit_tb_urls:
        tb_updated_url = tb_url.replace(".nii.gz", "--preparedtoseg.nii.gz")
        yazdan.ATTR.prepare_tb_to_rib_seg(tb_url, target_url=tb_updated_url,
                                          max_percentile = max_percentile,
                                          template_image_url = os.path.join(os.path.dirname(model_directory), "template-image.nii.gz"),
                                          )
        list_tb_updated_urls.append(tb_updated_url)
    list_available_models = os_sorted(glob(os.path.join(model_directory, "*Model-full.pth")))
    model_url = list_available_models[model_index]
    inference_results= yazdan.DL.model_inference_segmentation(model_url = model_url,
                                                              list_images = list_tb_updated_urls,
                                                              predict_directory = predict_directory,
                                                              organ_name = organ_name,
                                                              save_results = True,
                                                              prefix = prefix, 
                                                              suffix = suffix,
                                                              Remove_first_dim = Remove_first_dim,
                                                              keep_largest_segment = keep_largest_segment, 
                                                              num_largest_segment = num_largest_segment,
                                                              )
    return inference_results

def prepare_external_data_step_1(tb_original_url):
    import SimpleITK as sitk
    import numpy as np
    import matplotlib.image
    import os
    import yazdan
    def UpdateInfo(image):
        image.SetSpacing([x for x in image_joint_foregroud.GetSpacing() if x !=1])
        return image
    # both image
    image_both = sitk.ReadImage(tb_original_url)
    # image_both = sitk.DICOMOrient(image_both, "LAS")
    
    # single view images
    if image_both.GetDirection()[4] == -1: 
        image_ant = image_both[:,:,0]
        image_post = image_both[:,:,1]
    else:
        image_ant = image_both[:,:,1]
        image_post = image_both[:,:,0]
        
    array_ant = sitk.GetArrayFromImage(image_ant)
    array_post = sitk.GetArrayFromImage(image_post)
    array_post_flip = np.flip(array_post, axis=1)
    tb_joint_array = np.concatenate((array_ant[np.newaxis, :,:], array_post_flip[np.newaxis, :,:]), axis = 0)
    
    tb_joint_image = sitk.GetImageFromArray(tb_joint_array)
    tb_joint_image.CopyInformation(image_both)
    sitk.WriteImage(tb_joint_image, os.path.join(os.path.dirname(tb_original_url), "attr-tb-matched.nii.gz"))
    url_joint = os.path.join(os.path.dirname(tb_original_url), "attr-tb-matched.nii.gz")
    sitk.WriteImage(image_ant, os.path.join(os.path.dirname(tb_original_url), "extracted-anterior.nii.gz"))
    sitk.WriteImage(image_post, os.path.join(os.path.dirname(tb_original_url), "extracted-posterior.nii.gz"))
    #separate-image-ant-post
    annon_id = url_joint.split("\\")[-2]
    image_joint = sitk.ReadImage(url_joint)
    image_joint = sitk.DICOMOrient(image_joint, "LAI")
    median, mask, image_joint_foregroud = yazdan.image.CropBckg(image_joint, treshold=(0,1))
    sorted_image_dims = yazdan.image.sort_dimensions_by_size(image_joint_foregroud)
    if sorted_image_dims[0] == 0:
        image_anterior = image_joint_foregroud[1,:,:]
        image_posterior = image_joint_foregroud[0,:,:]
    elif sorted_image_dims[0] == 1:
        image_anterior = image_joint_foregroud[:,1,:]
        image_posterior = image_joint_foregroud[:,0,:] 
    elif sorted_image_dims[0] == 2:
        image_anterior = image_joint_foregroud[:,:,1]
        image_posterior = image_joint_foregroud[:,:,0]
            
    array_anterior = sitk.GetArrayFromImage(image_anterior)
    array_posterior = sitk.GetArrayFromImage(image_posterior)
    
    matplotlib.image.imsave(os.path.join(os.path.dirname(tb_original_url),f"{annon_id} -- anterior.jpeg"), array_anterior)
    image_anterior_to_write = sitk.GetImageFromArray(array_anterior)
    
    matplotlib.image.imsave(os.path.join(os.path.dirname(tb_original_url),f"{annon_id} -- posterior.jpeg"), array_posterior)
    image_posterior_to_write = sitk.GetImageFromArray(array_posterior)
        
    summation_array = array_anterior + array_posterior
    matplotlib.image.imsave(os.path.join(os.path.dirname(tb_original_url),f"{annon_id}--summation.jpeg"), summation_array)
    
    summation_image = sitk.GetImageFromArray(summation_array)
    image_anterior_to_write.GetSpacing()
    
    
    image_anterior_to_write = UpdateInfo(image_anterior_to_write)
    image_posterior_to_write = UpdateInfo(image_posterior_to_write)
    summation_image = UpdateInfo(summation_image)
    
    sitk.WriteImage(image_anterior_to_write, url_joint.replace("attr-tb-matched.nii.gz", "anterior-true.nii.gz"))
    sitk.WriteImage(image_posterior_to_write, url_joint.replace("attr-tb-matched.nii.gz", "posterior-true.nii.gz"))
    sitk.WriteImage(summation_image, url_joint.replace("attr-tb-matched.nii.gz", "summation.nii.gz"))
    return url_joint.replace("attr-tb-matched.nii.gz", "summation.nii.gz"), url_joint.replace("attr-tb-matched.nii.gz", "anterior-true.nii.gz"), url_joint.replace("attr-tb-matched.nii.gz", "posterior-true.nii.gz")
def prepare_external_data_step_1_spot_view(tb_original_url):
    import SimpleITK as sitk
    import numpy as np
    import matplotlib.image
    import os
    import yazdan
    def UpdateInfo(image):
        image.SetSpacing([x for x in image_joint_foregroud.GetSpacing() if x !=1])
        return image
    # both image
    image_both = sitk.ReadImage(tb_original_url)
    # image_both = sitk.DICOMOrient(image_both, "LAS")
    
    # single view images
    if image_both.GetDimension()>3:
        image_both = image_both[:,:,:,0]
    if image_both.GetDirection()[4] == 1: 
        image_ant = image_both[:,:,1]
        image_post = image_both[:,:,0]
    else:
        image_ant = image_both[:,:,0]
        image_post = image_both[:,:,1]
        
    array_ant = sitk.GetArrayFromImage(image_ant)
    array_post = sitk.GetArrayFromImage(image_post)
    array_post_flip = np.flip(array_post, axis=1)
    tb_joint_array = np.concatenate((array_ant[np.newaxis, :,:], array_post_flip[np.newaxis, :,:]), axis = 0)
    
    tb_joint_image = sitk.GetImageFromArray(tb_joint_array)
    tb_joint_image.CopyInformation(image_both)
    sitk.WriteImage(tb_joint_image, os.path.join(os.path.dirname(tb_original_url), "attr-tb-matched.nii.gz"))
    url_joint = os.path.join(os.path.dirname(tb_original_url), "attr-tb-matched.nii.gz")
    sitk.WriteImage(image_ant, os.path.join(os.path.dirname(tb_original_url), "extracted-anterior.nii.gz"))
    sitk.WriteImage(image_post, os.path.join(os.path.dirname(tb_original_url), "extracted-posterior.nii.gz"))
    #separate-image-ant-post
    annon_id = url_joint.split("\\")[-2]
    image_joint = sitk.ReadImage(url_joint)
    image_joint = sitk.DICOMOrient(image_joint, "LAI")
    median, mask, image_joint_foregroud = yazdan.image.CropBckg(image_joint, treshold=(0,1))
    sorted_image_dims = yazdan.image.sort_dimensions_by_size(image_joint_foregroud)
    if sorted_image_dims[0] == 0:
        image_anterior = image_joint_foregroud[1,:,:]
        image_posterior = image_joint_foregroud[0,:,:]
    elif sorted_image_dims[0] == 1:
        image_anterior = image_joint_foregroud[:,1,:]
        image_posterior = image_joint_foregroud[:,0,:] 
    elif sorted_image_dims[0] == 2:
        image_anterior = image_joint_foregroud[:,:,1]
        image_posterior = image_joint_foregroud[:,:,0]
            
    array_anterior = sitk.GetArrayFromImage(image_anterior)
    array_posterior = sitk.GetArrayFromImage(image_posterior)
    
    matplotlib.image.imsave(os.path.join(os.path.dirname(tb_original_url),f"{annon_id} -- anterior.jpeg"), array_anterior)
    image_anterior_to_write = sitk.GetImageFromArray(array_anterior)
    
    matplotlib.image.imsave(os.path.join(os.path.dirname(tb_original_url),f"{annon_id} -- posterior.jpeg"), array_posterior)
    image_posterior_to_write = sitk.GetImageFromArray(array_posterior)
        
    summation_array = array_anterior + array_posterior
    matplotlib.image.imsave(os.path.join(os.path.dirname(tb_original_url),f"{annon_id}--summation.jpeg"), summation_array)
    
    summation_image = sitk.GetImageFromArray(summation_array)
    image_anterior_to_write.GetSpacing()
    
    
    image_anterior_to_write = UpdateInfo(image_anterior_to_write)
    image_posterior_to_write = UpdateInfo(image_posterior_to_write)
    summation_image = UpdateInfo(summation_image)
    
    sitk.WriteImage(image_anterior_to_write, url_joint.replace("attr-tb-matched.nii.gz", "anterior-true.nii.gz"))
    sitk.WriteImage(image_posterior_to_write, url_joint.replace("attr-tb-matched.nii.gz", "posterior-true.nii.gz"))
    sitk.WriteImage(summation_image, url_joint.replace("attr-tb-matched.nii.gz", "summation.nii.gz"))
    return url_joint.replace("attr-tb-matched.nii.gz", "summation.nii.gz"), url_joint.replace("attr-tb-matched.nii.gz", "anterior-true.nii.gz"), url_joint.replace("attr-tb-matched.nii.gz", "posterior-true.nii.gz")

def prepare_external_data_step_2(lsit_tb_original_urls, segmentation_model_directories = r"H:\ATTR\AI-Tasks\Task0_WBPlanarSegmentation\dataset-prepared-registration\model_directory",
                                 ):
    import os
    from glob import glob
    from tqdm import tqdm
    import yazdan
    import SimpleITK as sitk
    import numpy as np
    from termcolor import cprint
    lsit_tb_urls = []
    for tb_original_url in tqdm(lsit_tb_original_urls, desc = "Preparing-summation-image"):
        try:
            summation_url, anterior_url, posterior_url = yazdan.ATTR.prepare_external_data_step_1(tb_original_url)
            lsit_tb_urls.append(summation_url)
        except RuntimeError:
            summation_url, anterior_url, posterior_url = yazdan.ATTR.prepare_external_data_step_1_spot_view(tb_original_url)
            lsit_tb_urls.append(summation_url)
            cprint(f"spot view static mode was detrected!       {tb_original_url}", "grey", "on_red")
        except:
            cprint(f"this case was not processed!       {tb_original_url}", "grey", "on_blue")
        

    # Ribs
    inference_results = yazdan.ATTR.TB_rib_segment(lsit_tb_urls,
                                                   model_directory = os.path.join(segmentation_model_directories, "TB_Ribs"),
                                                   prefix = "",
                                                   suffix = "",
                                                   model_index = 0,
                                                   max_percentile = 99., 
                                                   keep_largest_segment = True,
                                                   num_largest_segment = 2,
                                                   )
    # LV
    inference_results = yazdan.ATTR.TB_rib_segment(lsit_tb_urls,
                                                   model_directory = os.path.join(segmentation_model_directories, "TB_LV"),
                                                   prefix = "",
                                                   suffix = "",
                                                   model_index = 0,
                                                   max_percentile = 99., 
                                                   keep_largest_segment = True,
                                                   num_largest_segment = 1,
                                                   )
    # Whole Heart
    inference_results = yazdan.ATTR.TB_rib_segment(lsit_tb_urls,
                                                   model_directory = os.path.join(segmentation_model_directories, "TB_WholeHeart"),
                                                   prefix = "",
                                                   suffix = "",
                                                   model_index = 0,
                                                   max_percentile = 99., 
                                                   keep_largest_segment = True,
                                                   num_largest_segment = 1,
                                                   )

    list_anterior_crop_to_LV = []
    list_anterior_crop_to_ribs= []
    list_anterior_crop_to_WholeHeart = []
    
    list_posterior_crop_to_LV = []
    list_posterior_crop_to_ribs= []
    list_posterior_crop_to_WholeHeart = []
    
    list_summation_crop_to_LV = []
    list_summation_crop_to_ribs= []
    list_summation_crop_to_WholeHeart = []
    
    for summation_url in tqdm(lsit_tb_urls, desc = "Croppping"):
        summation_prepared_url = summation_url.replace(".nii.gz", "--preparedtoseg_noClip.nii.gz")
        wholeheart_url = summation_prepared_url.replace("_noClip.nii.gz", "-TB_WholeHeart.nii.gz")
        ribs_url = summation_prepared_url.replace("_noClip.nii.gz", "-TB_Ribs.nii.gz")
        LV_url = summation_prepared_url.replace("_noClip.nii.gz", "-TB_LV.nii.gz")
        anterior_url = summation_prepared_url.replace("summation--preparedtoseg_noClip.nii.gz", "anterior-true.nii.gz")
        posterior_url = summation_prepared_url.replace("summation--preparedtoseg_noClip.nii.gz", "posterior-true.nii.gz")
        
        
        
        posterior_image = yazdan.image.add_3rd_dimension(posterior_url, wholeheart_url, 0)
        posterior_image = sitk.DICOMOrient(posterior_image, "LPS")
        posterior_image_norm = yazdan.image.sitk_rescale(posterior_image, input_max=np.percentile(sitk.GetArrayFromImage(posterior_image), 99.))[1]
        
        anterior_image = yazdan.image.add_3rd_dimension(anterior_url, wholeheart_url, 0)
        anterior_image = sitk.DICOMOrient(anterior_image, "LPS")
        anterior_image_norm = yazdan.image.sitk_rescale(anterior_image, input_max=np.percentile(sitk.GetArrayFromImage(anterior_image), 99.))[1]
        
        summation_crop_to_wholeheart = yazdan.image.crop_image_to_segment(image = summation_prepared_url,
                                                                  segment = wholeheart_url, 
                                                                  margin_mm = 50,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        posterior_crop_to_wholeheart = yazdan.image.crop_image_to_segment(image = posterior_image_norm,
                                                                  segment = wholeheart_url, 
                                                                  margin_mm = 50,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        anterior_crop_to_wholeheart = yazdan.image.crop_image_to_segment(image = anterior_image_norm,
                                                                  segment = wholeheart_url, 
                                                                  margin_mm = 50,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        summation_crop_to_LV = yazdan.image.crop_image_to_segment(image = summation_prepared_url,
                                                                  segment = LV_url, 
                                                                  margin_mm = 90,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        posterior_crop_to_LV = yazdan.image.crop_image_to_segment(image = posterior_image_norm,
                                                                  segment = LV_url, 
                                                                  margin_mm = 90,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        anterior_crop_to_LV = yazdan.image.crop_image_to_segment(image = anterior_image_norm,
                                                                  segment = LV_url, 
                                                                  margin_mm = 90,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        summation_crop_to_ribs = yazdan.image.crop_image_to_segment(image = summation_prepared_url,
                                                                  segment = ribs_url, 
                                                                  margin_mm = 0,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        posterior_crop_to_ribs = yazdan.image.crop_image_to_segment(image = posterior_image_norm,
                                                                  segment = ribs_url, 
                                                                  margin_mm = 0,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        anterior_crop_to_ribs = yazdan.image.crop_image_to_segment(image = anterior_image_norm,
                                                                  segment = ribs_url, 
                                                                  margin_mm = 0,
                                                                  crop_dims = (0,2),
                                                                  )[0]
        
        original_folder = os.path.dirname(summation_url)
        sitk.WriteImage(summation_crop_to_wholeheart, os.path.join(original_folder, "summation-crop-to-wholeheart.nii.gz"))
        sitk.WriteImage(summation_crop_to_LV, os.path.join(original_folder, "summation-crop-to-LV.nii.gz"))
        sitk.WriteImage(summation_crop_to_ribs, os.path.join(original_folder, "summation-crop-to-ribs.nii.gz"))
        
        sitk.WriteImage(anterior_crop_to_wholeheart, os.path.join(original_folder, "anterior-crop-to-wholeheart.nii.gz"))
        sitk.WriteImage(anterior_crop_to_LV, os.path.join(original_folder, "anterior-crop-to-LV.nii.gz"))
        sitk.WriteImage(anterior_crop_to_ribs, os.path.join(original_folder, "anterior-crop-to-ribs.nii.gz"))
        
        sitk.WriteImage(posterior_crop_to_wholeheart, os.path.join(original_folder, "posterior-crop-to-wholeheart.nii.gz"))
        sitk.WriteImage(posterior_crop_to_LV, os.path.join(original_folder, "posterior-crop-to-LV.nii.gz"))
        sitk.WriteImage(posterior_crop_to_ribs, os.path.join(original_folder, "posterior-crop-to-ribs.nii.gz"))
        
        
        list_anterior_crop_to_LV.append(os.path.join(original_folder, "anterior-crop-to-LV.nii.gz"))
        list_anterior_crop_to_ribs.append(os.path.join(original_folder, "anterior-crop-to-ribs.nii.gz"))
        list_anterior_crop_to_WholeHeart.append(os.path.join(original_folder, "anterior-crop-to-wholeheart.nii.gz"))
    
        list_posterior_crop_to_LV.append(os.path.join(original_folder, "posterior-crop-to-LV.nii.gz"))
        list_posterior_crop_to_ribs.append(os.path.join(original_folder, "posterior-crop-to-ribs.nii.gz"))
        list_posterior_crop_to_WholeHeart.append(os.path.join(original_folder, "posterior-crop-to-wholeheart.nii.gz"))
    
        list_summation_crop_to_LV.append(os.path.join(original_folder, "summation-crop-to-LV.nii.gz"))
        list_summation_crop_to_ribs.append(os.path.join(original_folder, "summation-crop-to-ribs.nii.gz"))
        list_summation_crop_to_WholeHeart.append(os.path.join(original_folder, "summation-crop-to-wholeheart.nii.gz"))
        prepeared_urls = {}
        prepeared_urls["list_anterior_crop_to_LV"] = list_anterior_crop_to_LV
        prepeared_urls["list_anterior_crop_to_ribs"] = list_anterior_crop_to_ribs
        prepeared_urls["list_anterior_crop_to_WholeHeart"] = list_anterior_crop_to_WholeHeart
        prepeared_urls["list_posterior_crop_to_LV"] = list_posterior_crop_to_LV
        prepeared_urls["list_posterior_crop_to_ribs"] = list_posterior_crop_to_ribs
        prepeared_urls["list_posterior_crop_to_WholeHeart"] = list_posterior_crop_to_WholeHeart
        prepeared_urls["list_summation_crop_to_LV"] = list_summation_crop_to_LV
        prepeared_urls["list_summation_crop_to_ribs"] = list_summation_crop_to_ribs
        prepeared_urls["list_summation_crop_to_WholeHeart"] = list_summation_crop_to_WholeHeart
        
    return prepeared_urls

def classify_external_single_model(list_images_test,
                      list_labels_test = "none",
                      task = "Detection-task",
                      crop_strategy= "crop-to-wholeheart",
                      input_image = "anterior.nii.gz",
                      predict_directory = "none", 
                      GradCam = False,
                      root_model_directory = r"C:\YazdanFiles\ATTR\AI-Tasks\Task1_WB_Planar_classification\model-directory",
                      create_plots = False,
                      ):
    import os
    import monai, torch
    from glob import glob
    from tqdm import tqdm
    import pandas as pd
    import SimpleITK as sitk
    from termcolor import cprint

    if list_labels_test == "none":
        list_labels_test = [1] * len(list_images_test)
        skip_evaluations = True
    else:
        skip_evaluations = False
    os.makedirs(predict_directory, exist_ok=True)
    patch_image_size = (96,96,1)
    model_network = "Densnet201-e-4"
    model_config_name = f"{crop_strategy}--{input_image.replace('.nii.gz', '')}"
    model_directory = os.path.join(root_model_directory, task, f"{model_network}--{crop_strategy}--{input_image.replace('.nii.gz', '')}-True")
    model = monai.networks.nets.densenet201(spatial_dims = 2, in_channels = 1, out_channels = 2)
    device = torch.device("cuda")

    data_loader_dict_test = yazdan.classification.data_loader_resize(list_images_test, list_labels_test,
                                                                image_size = patch_image_size,
                                                                cachedata = False,
                                                                orientation = "LSP",
                                                                clip = True,
                                                                batch_size = 1,
                                                                data_split = "test",
                                                                anti_aliasing = True,
                                                                input_lower_intensity = 0, 
                                                                input_upper_intensity = 1,
                                                                input_b_min = 0, 
                                                                input_b_max = 1, 
                                                                num_class = 2,
                                                                data_loader_type = "Thread",
                                                                data_loader_num_worker = 0,
                                                                scale_image_to_self_max = False,
                                                                ) 
    
    data_loader_test = data_loader_dict_test["data_loader"]
    post_pred = data_loader_dict_test["post_pred"]
    post_label = data_loader_dict_test["post_label"]
    
    
    list_models = glob(os.path.join(model_directory,"fold-*", "*.pth"))
    total_test_eval = []
    model.to(device)
    for model_url in list_models:
        model.eval()
        model.load_state_dict(torch.load(model_url))
        model_name = os.path.basename(model_url).replace(".pth", "")
        fold = model_url.split("\\")[-2]
        if GradCam:
            os.makedirs(os.path.join(predict_directory, "GradCam" + model_name), exist_ok = True)
        row = 1
        predicted_labels = []
        ground_truth_labels = []
        
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
                    
            y_pred_test = torch.tensor([], dtype=torch.float32, device=device)
            y_test = torch.tensor([], dtype=torch.long, device=device)
            list_test_names  = []
           
            for index, test_data in enumerate(tqdm(data_loader_test, desc = model_name, leave = True, position=0, total = len(data_loader_test), ncols = 150)):
                test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
                url_temp = test_data["image_meta_dict"]["filename_or_obj"][0]
                name_temp = os.path.basename(test_data["image_meta_dict"]["filename_or_obj"][0])
                
                test_images = torch.squeeze(test_images, dim = -1)
                test_outputs_prob = model(test_images)
                if GradCam:
                    try:
                        grad_images, titles, log_scales = yazdan.classification.GradCam(model, test_data)
                        input_image, Occ_sens_image, grad_cam_image = torch.squeeze(grad_images[0]).cpu().numpy(),torch.squeeze(grad_images[1]).cpu().numpy(), torch.squeeze(grad_images[2]).cpu().numpy()
                        """
                        temp to add
                        list_of_image_links = [cv2.cvtColor(normalize(input_image)*256,cv2.COLOR_GRAY2RGB), cv2.cvtColor(normalize(Occ_sens_image)*256,cv2.COLOR_GRAY2RGB), cv2.cvtColor(normalize(grad_cam_image)*256,cv2.COLOR_GRAY2RGB)]
                        yazdan.image.image_concat_grid(list_of_image_links, grid_size = (1,3), output_image_link = os.path.join(predict_directory,"GradCam", f"{name_temp}-{index}.jpeg"))
                        """
                        sitk.WriteImage(sitk.GetImageFromArray(input_image), 
                                        os.path.join(predict_directory, "GradCam" + model_name, f"{name_temp}-{index}-image.nii.gz"))
                        sitk.WriteImage(sitk.GetImageFromArray(Occ_sens_image), 
                                        os.path.join(predict_directory, "GradCam" + model_name, f"{name_temp}-{index}-Occ_sens_image.nii.gz"))
                        sitk.WriteImage(sitk.GetImageFromArray(grad_cam_image), 
                                        os.path.join(predict_directory, "GradCam" + model_name, f"{name_temp}-{index}-grad_cam_image.nii.gz"))
                    except:
                        print(url_temp + "no grad cam")
                        
                   
                
                y_pred_test = torch.cat([y_pred_test, test_outputs_prob], dim=0)
                y_test = torch.cat([y_test, test_labels], dim=0)
                
                test_outputs = test_outputs_prob.argmax(dim=1)  
                soft_max_torch = torch.nn.Softmax()
                probabilities_value = soft_max_torch(test_outputs_prob) 
                y_onehot = post_label(test_outputs)
                y_pred_act = post_pred(y_onehot)
    
                value_temp = torch.eq(test_outputs, test_labels)
                metric_count += len(value_temp)
                num_correct += value_temp.sum().item()
                
                list_test_names.append(url_temp)
                predicted_labels.append(test_outputs.cpu().data.numpy()[0])
                ground_truth_labels.append(test_labels.cpu().data.numpy()[0])
        
                row += 1
            metric = num_correct / metric_count
            print("evaluation metric:", metric)
            ######
            """y_test is ground truth
            y_pred_test i s predicted weigth
            y_pred_act_test is activated prediction
            y_pred_slected_class is the predicted class
            """
            y_onehot_test = [post_label(i) for i in monai.data.decollate_batch(y_test, detach=False)]
            y_pred_act_test = [post_pred(i) for i in monai.data.decollate_batch(y_pred_test)]
            y_pred_slected_class = [int(x.argmax().cpu().data.numpy()) for x in y_pred_act_test]
            y_ground_truth_class = [int(x.argmax().cpu().data.numpy()) for x in y_onehot_test]
            predicted_probailities = [x.cpu().data.numpy().astype(float) for x in y_pred_act_test]
            if create_plots:
                try:
                    yazdan.classification.ROC_fancy_plot(predicted_probailities, y_onehot_test, 
                                                         output_url=os.path.join(predict_directory, f"{model_config_name}--{model_name}--{fold}" + "-Fancy-ROC.png"), 
                                                         show = True)                    
                except:
                    pass
                yazdan.classification.ROC_Plot(y_onehot_test, predicted_probailities, os.path.join(predict_directory, f"{model_config_name}--{model_name}--Fold {fold}" + "-ROC.png"))
                ouput_url_confusion_matrix = os.path.join(predict_directory, f"{model_config_name}--{model_name}--{fold}" + "-confusion.png")
                try:
                    yazdan.classification.ConfusionMatrix(y_ground_truth_class, y_pred_slected_class,
                                                          ouput_url = ouput_url_confusion_matrix, 
                                                          dpi = 1000,
                                                          normalize = True,
                                                          )
                except:
                    cprint("Error Happened at Creating Confusion Matrix")
                
            predicted_probailities_dict = {}
            for class_num in range(len(predicted_probailities[0])):
                predicted_probailities_dict[f"class-{class_num}-Prob"] = [x[class_num] for x in predicted_probailities]
            
            ResultsThisModel = {"names" : list_test_names, "GroundTruthLabel" : y_ground_truth_class, 
                                "PredictedLabel" : y_pred_slected_class, 
                                }
            ResultsThisModel.update(predicted_probailities_dict)
            ResultsThisModel_df = pd.DataFrame(ResultsThisModel)
            ResultsThisModel_df.to_excel(os.path.join(predict_directory, f"{model_config_name}--{model_name}--{fold}" + ".xlsx"))
            
            y_onehot_test = [post_label(i) for i in monai.data.decollate_batch(y_test, detach=False)]
            y_pred_act_test = [post_pred(i) for i in monai.data.decollate_batch(y_pred_test)]
            model_test_eval = {}
            model_test_eval["model_name"]  =  model_name
            model_classification_eval = yazdan.classification.classification_eval(y_pred_act_test, y_onehot_test)[0]
            model_test_eval.update(model_classification_eval)
            total_test_eval.append(model_test_eval)
    total_test_eval_df = pd.DataFrame(total_test_eval)

    total_test_eval_df.to_excel(os.path.join(predict_directory, "whole-evaluations.xlsx"))    
    return total_test_eval_df, predicted_labels, predicted_probailities

def ensemble_folds(pipeline_url, metric = "last_epoch_model"):
    import os
    from natsort import os_sorted
    from glob import glob
    import pandas as pd
    list_metric_all_flods = os_sorted(glob(os.path.join(pipeline_url, f"*{metric}*--fold-*.xlsx")))
    single_model_all_folds_df = pd.DataFrame()
    single_model_average_df = pd.DataFrame()
    for single_model_url in list_metric_all_flods:
        model_df = pd.read_excel(single_model_url)
        model_df["id"] = [x.split('\\')[-2] for x in model_df["names"]]
        model_df = model_df.sort_values(by = "id")
        model_df = model_df.reset_index()
        single_model_all_folds_df = pd.concat([single_model_all_folds_df, model_df], axis = 1)
        
    single_model_average_df["url"] = model_df["names"] 
    single_model_average_df["id"] = model_df["id"] 
    single_model_average_df["metric"] = metric
    single_model_average_df["class-0-Prob"] = single_model_all_folds_df[["class-0-Prob"]].mean(axis=1)
    single_model_average_df["class-1-Prob"] = single_model_all_folds_df[["class-1-Prob"]].mean(axis=1)
    single_model_average_df.to_excel(os.path.join(pipeline_url, f"{metric}--average-three-folds.xlsx"))

    return single_model_average_df, os.path.join(pipeline_url, f"{metric}--average-three-folds.xlsx")


def ensemble_all_crops(classification_out_url, input_image = "allinputs", task = "Detection-task", metric = "last_epoch_model"):
    import os
    from natsort import os_sorted
    from glob import glob
    import pandas as pd
    from termcolor import cprint
    if input_image == "allinputs":
        list_task = os_sorted(glob(os.path.join(classification_out_url, f"*{task}")))
    else:
        list_task = os_sorted(glob(os.path.join(classification_out_url, f"{input_image}*{task}")))
    
    ensebmbled_df = pd.DataFrame()   
    if not len(list_task):
        cprint(f"\nNo output was found for input:  {input_image}, task: {task}!!!", "white", "on_blue")
        return ensebmbled_df
    
    all_crop_ensembled_df = pd.DataFrame()
    for pipeline_detection_url in list_task:
        pipeline_url = pipeline_detection_url
        metric="last_epoch_model"
        ensemble_df, ensemble_url = yazdan.ATTR.ensemble_folds(pipeline_detection_url, metric = metric)
        all_crop_ensembled_df = pd.concat([all_crop_ensembled_df, ensemble_df])
    
    ensebmbled_df["class-0-Prob"] = all_crop_ensembled_df[["class-0-Prob"]].mean(axis=1)
    ensebmbled_df["class-1-Prob"] = all_crop_ensembled_df[["class-1-Prob"]].mean(axis=1)
    ensebmbled_df["url"] = all_crop_ensembled_df["url"]
    ensebmbled_df["id"] = all_crop_ensembled_df["id"]
    ensebmbled_df["input_image"] = input_image
    ensebmbled_df["crop-strategy"] =  [os.path.basename(x).replace(".nii.gz", "") for x in all_crop_ensembled_df["url"]]
    ensebmbled_df["task"] = task
    ensebmbled_df = ensebmbled_df[["url", "id", "input_image","task","crop-strategy", "class-0-Prob", "class-1-Prob"]]
    ensebmbled_df.to_excel(os.path.join(classification_out_url, f"{input_image}--{task}-Ensembled.xlsx"))
    return ensebmbled_df

def external_classify_whole(lsit_tb_original_urls, 
                            target, 
                            model_path_on_your_device,
                            list_labels_test = "none",
                            metric = "last_epoch_model"
                            ):
    import yazdan
    import os
    from termcolor import cprint
    from tqdm import tqdm
    import itertools
    import pandas as pd
    
    list_preprocessed_names = ['anterior-crop-to-LV.nii.gz',
     'anterior-crop-to-ribs.nii.gz',
     'anterior-crop-to-wholeheart.nii.gz',
     'anterior-true.nii.gz',
     'attr-tb-matched.nii.gz',
     'extracted-anterior.nii.gz',
     'extracted-posterior.nii.gz',
     'posterior-crop-to-LV.nii.gz',
     'posterior-crop-to-ribs.nii.gz',
     'posterior-crop-to-wholeheart.nii.gz',
     'posterior-true.nii.gz',
     'summation--preparedtoseg-TB_LV--probability.nii.gz',
     'summation--preparedtoseg-TB_LV.nii.gz',
     'summation--preparedtoseg-TB_Ribs--probability.nii.gz',
     'summation--preparedtoseg-TB_Ribs.nii.gz',
     'summation--preparedtoseg-TB_WholeHeart--probability.nii.gz',
     'summation--preparedtoseg-TB_WholeHeart.nii.gz',
     'summation--preparedtoseg.nii.gz',
     'summation--preparedtoseg_noClip.nii.gz',
     'summation-crop-to-LV.nii.gz',
     'summation-crop-to-ribs.nii.gz',
     'summation-crop-to-wholeheart.nii.gz',
     'summation.nii.gz']
    classification_model_dir_root = os.path.join(model_path_on_your_device,  "classification")
    segmentation_model_directories = os.path.join(model_path_on_your_device,  "segmentations")
    lsit_tb_original_urls = [x for x in lsit_tb_original_urls if not os.path.basename(x) in list_preprocessed_names]
    cprint("\n\n Preparing Datsets for Segmentation\n\n", "white" , "on_cyan")
    prepeared_urls = yazdan.ATTR.prepare_external_data_step_2(lsit_tb_original_urls, segmentation_model_directories = segmentation_model_directories)
    available_pipelines = prepeared_urls.keys()
    cprint("\n\n Classification \n\n", "white" , "on_cyan")
    for pipeline in tqdm(available_pipelines, colour = "Red", desc = "PipeLine Progress"):
        input_image = pipeline.split("_")[1] + ".nii.gz"
        crop_strategy = '-'.join(pipeline.split("_")[2:])
        pipeline
        for task in ["Detection-task", "Severity-task-2classes"]:
            predict_directory = os.path.join(target, f'{pipeline.replace("list_","")}-{task}')
            yazdan.ATTR.classify_external_single_model(list_images_test = prepeared_urls[pipeline],
                                  list_labels_test = list_labels_test,
                                  task = task,
                                  crop_strategy= crop_strategy,
                                  input_image = input_image,
                                  predict_directory = predict_directory,
                                  GradCam = False,
                                  root_model_directory = classification_model_dir_root,
                                  )
        
    
    cprint("\n\n Ensembling methods \n\n", "white" , "on_cyan")
    final_all_decision =  pd.DataFrame()
    for input_image,task in itertools.product(["anterior", "posterior", "summation", "allinputs"], ["Detection-task", "Severity-task-2classes"]):
        ensemble_decision = yazdan.ATTR.ensemble_all_crops(classification_out_url = target,
                                                           input_image =input_image,
                                                           task = task, 
                                                           metric = metric,
                                                           )
        final_all_decision = pd.concat([final_all_decision, ensemble_decision])
        
    final_all_decision["decision"] =  (final_all_decision['class-1-Prob'] > final_all_decision['class-0-Prob']).astype(int)
    final_all_decision.to_excel(os.path.join(target, "whole-decision-classification.xlsx"))
    return available_pipelines, ensemble_decision
      
