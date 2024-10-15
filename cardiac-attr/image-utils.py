def get_orientation(image):
    import nibabel as nib
    if isinstance(image, str):
        image = nib.load(image)
    orientation = nib.aff2axcodes(image.affine)
    return orientation
def select_last_dimensions(image_array, num_dims = 3):
    while len(image_array.shape) > 3:
        image_array = image_array[0,]
    return image_array
def CropBckg(image, treshold = "otsu"):
    import SimpleITK as sitk
    '''
    threshold_based_crop_and_bg_median was original name
    link : https://simpleitk.org/SPIE2019_COURSE/03_data_augmentation.html
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box and compute the background 
    median intensity.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
        Background median intensity value.
    '''
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    inside_value = 0
    outside_value = 255
    if treshold == "otsu":
        bin_image = sitk.OtsuThreshold(image, inside_value, outside_value)
    else:
        bin_image = sitk.BinaryThreshold(image, lowerThreshold=treshold[0], upperThreshold=treshold[1], insideValue=inside_value, outsideValue=outside_value)

    # Get the median background intensity
    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats_filter.SetBackgroundValue(outside_value)
    label_intensity_stats_filter.Execute(bin_image,image)
    bg_mean = label_intensity_stats_filter.GetMedian(inside_value)
    
    # Get the bounding box of the anatomy
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()    
    label_shape_filter.Execute(bin_image)
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return bg_mean, bin_image>1, sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
def sitk_rescale(image, input_min = "image-min", input_max = "image-max", output_min = 0, output_max = 1):
    import SimpleITK as sitk
    import numpy as np
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    image_array = sitk.GetArrayFromImage(image)
    if input_min == "image-min":
        input_min = np.min(image_array)
    if input_max == "image-max":
        input_max = np.max(image_array)
        
    array_no_clip = (image_array - input_min) / (input_max - input_min)
    array_no_clip = output_min + (array_no_clip * (output_max - output_min))
    image_no_clip = sitk.GetImageFromArray(array_no_clip)
    image_no_clip = yazdan.image.CopyInfo(ReferenceImage = image, UpdatingImage = image_no_clip)
    
    array_clip = np.interp(image_array, (input_min, input_max), (output_min, output_max))
    image_clip = sitk.GetImageFromArray(array_clip)
    image_clip = yazdan.image.CopyInfo(ReferenceImage = image, UpdatingImage = image_clip)
    return image_clip, image_no_clip    
def CopyInfo(ReferenceImage, UpdatingImage, origin = True, spacing = True, direction = True):
    import SimpleITK as sitk
    if isinstance(ReferenceImage, str):
        ReferenceImage = sitk.ReadImage(ReferenceImage)
    if isinstance(UpdatingImage, str):
        UpdatingImage = sitk.ReadImage(UpdatingImage)
    UpdatedImage = UpdatingImage 
    if origin:
        UpdatedImage.SetOrigin(ReferenceImage.GetOrigin())
    if spacing:
        UpdatedImage.SetSpacing(ReferenceImage.GetSpacing())
    if direction:
        UpdatedImage.SetDirection(ReferenceImage.GetDirection())
    return UpdatedImage
def keep_largest_segments(segment, engine = "sitk", numbner_of_objects = 1):
    import SimpleITK as sitk
    import numpy as np
    segment_url = segment
    
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
        segment = sitk.Cast(segment, sitk.sitkUInt8)
    if "-TB_LV.nii.gz" in segment_url or "-TB_WholeHeart.nii.gz" in segment_url:
        "the ribs were segmented befire, using them to crop the model output"
        ribs_seg_url = segment_url.replace("-TB_LV.nii.gz", "-TB_Ribs.nii.gz").replace("-TB_WholeHeart.nii.gz", "-TB_Ribs.nii.gz")
        ribs_segment = sitk.ReadImage(ribs_seg_url)
        crop_information = yazdan.image.crop_image_to_segment(ribs_segment, ribs_segment)[-1]
        segment_updated = segment
        segment_updated[:, :crop_information["crop_start_indices"][1]] = 0
        segment_updated[:, crop_information["crop_end_indices"][1]: ] = 0
        
        if len(np.unique(sitk.GetArrayFromImage(segment_updated))) > 1:
            segment = segment_updated
            
    if engine == "sitk":
        component_image = sitk.ConnectedComponent(segment)
        sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
        # largest_component_binary_image = sorted_component_image == 1
        largest_component_binary_image = sum([sorted_component_image == label for label in range(1, numbner_of_objects + 1)])
    elif engine == "monai":
        from monai.transforms import KeepLargestConnectedComponent
        segment_array = sitk.GetArrayFromImage(segment)
        keep_large_transform = KeepLargestConnectedComponent(num_components = numbner_of_objects)
        largest_component_binary_array = keep_large_transform(segment_array)
        largest_component_binary_image = sitk.GetImageFromArray(largest_component_binary_array)
    return largest_component_binary_image
def sort_dimensions_by_size(image):
    image_Size = image.GetSize()
    image_Size_list_sorted = sorted([(value, index) for index, value in enumerate(image_Size)])
    sorted_values = [value for value, _ in image_Size_list_sorted]
    sorted_dimension_indices = [index for _, index in image_Size_list_sorted]
    return sorted_dimension_indices
def add_3rd_dimension(image_url, template_image_url, axis = 1):
    import SimpleITK as sitk
    import numpy as np
    
    if isinstance(template_image_url, str):
        template_image = sitk.ReadImage(template_image_url)
    elif isinstance(template_image_url, sitk.Image):
        template_image = template_image_url

        
    if isinstance(image_url, str):
        image = sitk.ReadImage(image_url)
        image_array = sitk.GetArrayFromImage(image)
    elif isinstance(image, sitk.Image):
        image = image_url
        image_array = sitk.GetArrayFromImage(image)
        
    array_expanded = np.expand_dims(image_array, axis = axis)
    image_expanded = sitk.GetImageFromArray(array_expanded)
    
    
    image_expanded.SetSpacing((template_image.GetSpacing()[2],  template_image.GetSpacing()[1], template_image.GetSpacing()[0]))
    image_expanded.SetDirection(template_image.GetDirection())
    return image_expanded
def crop_image_to_segment(image, segment, crop_dims = "all", margin_mm = 0, 
                          lowerThreshold = 0.1, upperThreshold = .9,
                          insideValue = 0, outsideValue =1, 
                          force_match = False
                          ):
    import SimpleITK as sitk
    from termcolor import cprint
    if isinstance(image, str):
        image = sitk.ReadImage(image)
        image = sitk.DICOMOrient(image, "LPS")
    if isinstance(segment, str):
        segment = sitk.ReadImage(segment)
        segment = sitk.DICOMOrient(segment, "LPS")
        # finding crop area
    if force_match:
        segment = yazdan.image.match_space(input_image = segment, reference_image = image)
    segment = sitk.Cast(segment, sitk.sitkUInt8)
    segment_non_binary = segment
    segment = sitk.BinaryThreshold(segment, lowerThreshold=lowerThreshold, 
                                   upperThreshold=upperThreshold, 
                                   insideValue = insideValue, outsideValue = outsideValue)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(segment)
    bounding_box = label_shape_filter.GetBoundingBox(1) 

    start_physical_point = segment.TransformIndexToPhysicalPoint(bounding_box[0 : int(len(bounding_box) / 2)])
    end_physical_point = segment.TransformIndexToPhysicalPoint([x+sz for x,sz in zip(bounding_box[0 : int(len(bounding_box) / 2)], bounding_box[int(len(bounding_box) / 2) :])])
    if any([start>end for start, end in zip(start_physical_point, end_physical_point)]):
        cprint("warning directions have issues, check the output !!!!!", "white", "on_red")
    
    start_physical_point = [x - margin_mm for x in start_physical_point]
    end_physical_point = [x + margin_mm for x in end_physical_point]
    # crop using the indexes
    image_crop_start_indices = image.TransformPhysicalPointToIndex(start_physical_point)
    image_crop_end_indices = image.TransformPhysicalPointToIndex(end_physical_point)
    
    segment_crop_start_indices = segment.TransformPhysicalPointToIndex(start_physical_point)
    segment_crop_end_indices = segment.TransformPhysicalPointToIndex(end_physical_point)
    
    
    image_crop_sizes = [a-b for a,b in zip(image_crop_end_indices , image_crop_start_indices)]
    segment_crop_sizes = [a-b for a,b in zip(segment_crop_end_indices , segment_crop_start_indices)]
            
    
    image_crop_start_indices = list(image_crop_start_indices)
    for dimension, image_crop_start_index in enumerate(image_crop_start_indices):
        if image_crop_start_index < 0 : 
            image_crop_start_indices[dimension] = 0

    image_crop_sizes = list(image_crop_sizes)
    for dimension, image_crop_size in enumerate(image_crop_sizes):
        if image_crop_size + image_crop_start_indices[dimension] > image.GetSize()[dimension]: 
            image_crop_sizes[dimension] = image.GetSize()[dimension] - image_crop_start_indices[dimension] -1
        
    segment_crop_start_indices = list(segment_crop_start_indices)
    for dimension, segment_crop_start_index in enumerate(segment_crop_start_indices):
        if segment_crop_start_index < 0 : 
            segment_crop_start_indices[dimension] = 0


    segment_crop_sizes = list(segment_crop_sizes)
    for dimension, segment_crop_size in enumerate(segment_crop_sizes):
        if segment_crop_size + segment_crop_start_indices[dimension] > segment.GetSize()[dimension]: 
            segment_crop_sizes[dimension] = segment.GetSize()[dimension] - segment_crop_start_indices[dimension] -1
            
    image_crop_start_indices = list(image_crop_start_indices)
    if crop_dims == "all":
        "do nothging -- crop in all dimension"
    else:
        no_crop_dims = [x for x in [0,1,2] if x not in crop_dims]
        for dimension in no_crop_dims:
            image_crop_start_indices[dimension] = 0
            image_crop_sizes[dimension] = image.GetSize()[dimension]
    
    image_cropped = sitk.RegionOfInterest(image, image_crop_sizes, image_crop_start_indices)
    segment_cropped = sitk.RegionOfInterest(segment, segment_crop_sizes, segment_crop_start_indices)
    segment_non_binary_cropped = sitk.RegionOfInterest(segment_non_binary, segment_crop_sizes, segment_crop_start_indices)
    crop_box_out = {}
    crop_box_out["start_physical_point"] = start_physical_point
    crop_box_out["end_physical_point"] = end_physical_point
    crop_box_out["crop_start_indices"] = image_crop_start_indices
    crop_box_out["crop_end_indices"] = image_crop_end_indices
    crop_box_out["crop_sizes"] = image_crop_sizes
    
    return image_cropped, segment_cropped, segment_non_binary_cropped, crop_box_out
