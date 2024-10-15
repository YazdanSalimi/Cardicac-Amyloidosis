def classification_eval(prediction_activated, groundtruth, 
                            weights = tuple([1 for x in range(1000)]),
                            is_binary = True,
                            ):

    import monai
    import torch
    import roc_utils as ru
    import numpy as np
    confusion_metrics = monai.metrics.ConfusionMatrixMetric(metric_name=["F1",
                                                                              "sensitivity",
                                                                              "specificity",
                                                                              "precision",
                                                                              "accuracy"],
                                                                 reduction="mean_batch")
    try:
        prediction_binary = [(x == x.max()).float() for x in prediction_activated]
    except:
        prediction_binary = [(x == x.max()) for x in prediction_activated]
        prediction_binary = torch.tensor(prediction_binary)
        groundtruth = torch.tensor(groundtruth)
        prediction_activated = torch.tensor(prediction_activated)
    
    confusion_metrics(prediction_binary, groundtruth)
    if is_binary:
        F1_score,sensitivity_score,specificity_score,precision_score,accuracy_score = [float(x.detach()[0]) for x in confusion_metrics.aggregate()]
    else:
        F1_score,sensitivity_score,specificity_score,precision_score,accuracy_score = [torch.nanmean(x).item() for x in confusion_metrics.aggregate()]
    ############
    F1_score_multi_class,sensitivity_score_multi_class,specificity_score_multi_class,precision_score_multi_class,accuracy_score_multi_class = [x.cpu().numpy() for x in confusion_metrics.aggregate()]
    
    if isinstance(prediction_activated, np.ndarray):
        if prediction_activated.ndim > 1:
            num_classes = prediction_activated.shape[1]
            AUC_multi_class = []
            for index_class in range(num_classes):
                AUC_multi_class.append(ru.compute_roc(X=prediction_activated[:,index_class], y=groundtruth[:,index_class], pos_label=True).auc)
        else:
            AUC_multi_class = 0  
    else:# isinstance(prediction_activated, list):
        num_classes = len(prediction_activated[0]) 
        if len(prediction_activated[0]) > 2:
            AUC_multi_class = []
            for index_class in range(num_classes):
                AUC_multi_class.append(ru.compute_roc(X = [x[index_class].cpu() for x in prediction_activated],
                                                      y = [x[index_class].cpu() for x in groundtruth],
                                                      pos_label=True).auc)
        else:
            AUC_multi_class = 0  
    ############
    ROC_AUC_mteric = monai.metrics.ROCAUCMetric()

    ROC_AUC_mteric(prediction_activated, groundtruth)
    AUC_metric = ROC_AUC_mteric.aggregate()
    
    classification_eval = {"F1_score" : F1_score, "sensitivity_score" : sensitivity_score, 
                           "specificity_score" : specificity_score, "precision_score" : precision_score, 
                           "accuracy_score" : accuracy_score,
                             "sens_spec_average" : (sensitivity_score + specificity_score) / 2,
                               "AUC_metric" : AUC_metric,
                               }
    multi_calssification_eval = {"F1_score" : F1_score_multi_class, "sensitivity_score" : sensitivity_score_multi_class, 
                           "specificity_score" : specificity_score_multi_class, "precision_score" : precision_score_multi_class, 
                           "accuracy_score" : accuracy_score_multi_class,
                             "sens_spec_average" : (sensitivity_score_multi_class + specificity_score_multi_class) / 2,
                               "AUC_metric" : AUC_multi_class,
                               }
    if num_classes>2:
        def weigted_average(array, weights_function = weights):
            if isinstance(array, int):
                return 0
            pure_array = [x for x in array if not np.isnan(x)]
            if not len(pure_array):
                average_weighted = 0
            else:
                average_weighted = np.average(pure_array, weights = weights_function[:len(pure_array)])
            return average_weighted
        weighted_classification_eval = {"F1_score" : weigted_average(F1_score_multi_class),
                                                                     "sensitivity_score" : weigted_average(sensitivity_score_multi_class), 
                               "specificity_score" : weigted_average(specificity_score_multi_class),
                                                                    "precision_score" : weigted_average(precision_score_multi_class), 
                               "accuracy_score" : weigted_average(accuracy_score_multi_class),
                                 "sens_spec_average" : (weigted_average(sensitivity_score_multi_class) + weigted_average(specificity_score_multi_class)) / 2,
                                   "AUC_metric" : weigted_average(AUC_multi_class),
                                   }
    else:
        weighted_classification_eval = classification_eval
            
    ROC_AUC_mteric.reset()
    confusion_metrics.reset()
    return multi_calssification_eval, classification_eval, weighted_classification_eval

def data_loader_resize(list_images, list_labels, 
                data_loader_type = "Thread",
                cachedata = True,
                cache_rate = 1,
                augmentation = False,
                transform = "default",
                batch_size = 2,
                image_size = (96,96,96),
                orientation = "RAS",
                input_interpolation = "bilinear",
                anti_aliasing = True,
                samrt_num_workers = 0,
                input_lower_intensity = -70,
                input_upper_intensity = 170,
                input_b_min = 0,
                input_b_max = 1,
                clip = False,
                data_split = "train",
                num_class = 2,
                ImageCropForeGroundMargin = 0,
                scale_image_to_self_max = True,
                data_loader_num_worker = 12,
                ):
    # list_images, list_labels, 
    # data_loader_type = "Thread"; cachedata = True;
    # cache_rate = 1; augmentation = False; transform = "default";
    # batch_size = 2; pixdim = (1,1,1); image_size = (96,96,96);
    # orientation = "RAS"; input_interpolation = "bilinear";
    # samrt_num_workers = 0;
    # input_lower_intensity = -70;
    # input_upper_intensity = 170;
    # input_b_min = 0; input_b_max = 1;
    # clip = False; task = "classification"; data_split = "train";
    # num_class = 2

    import monai
    import numpy as np
    import torch, yazdan
    from termcolor import cprint
    data_dict = [{"image": image, "label": label} for image, label in zip(list_images, list_labels)]
    num_class_labels = len(np.unique(list_labels))
    if num_class_labels != num_class:
        cprint(f"Warning!!!!  Number of labels are different , existed labels are : {np.unique(list_labels)}", "red", "on_cyan")
    # transformation
    if data_split == "train":
        pass
    else:
        initial_transform = monai.transforms.Compose(
                [
                    monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True, image_only=False),
                    monai.transforms.Orientationd(keys=["image"], axcodes = orientation),
                    monai.transforms.CropForegroundd(keys=["image"], source_key="image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                    monai.transforms.Resized(keys=["image"], spatial_size=image_size,
                                             mode = input_interpolation, 
                                             anti_aliasing=anti_aliasing),
                    monai.transforms.ScaleIntensityRanged(keys = ["image"],
                                                          a_min = input_lower_intensity, a_max = input_upper_intensity,
                                                          b_min = input_b_min, b_max=input_b_max, 
                                                          clip=clip) if not scale_image_to_self_max else monai.transforms.ScaleIntensityd(keys = ["image"],
                                                                                                minv = input_b_min, maxv=input_b_max),  
                ]
            )
    if transform != "default":
        initial_transform = transform
    post_pred = monai.transforms.Compose([monai.transforms.Activations(softmax=True)])
    post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(to_onehot = num_class)])
    # data loaders
    check_ds = monai.data.Dataset(data=data_dict, transform=initial_transform)
    check_loader = monai.data.DataLoader(check_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["image"].shape, check_data["label"])
    input_shape = check_data["image"].shape
    input_sample = yazdan.image.select_last_dimensions(check_data["image"], 3)
    # yazdan.image.view([input_sample])
    # create a data loader
    if cachedata:
        dataset = monai.data.CacheDataset(data=data_dict, transform=initial_transform, cache_rate = cache_rate)
    else:
        dataset = monai.data.Dataset(data=data_dict, transform=initial_transform)  
    if data_loader_type == "Thread":
        data_loader = monai.data.ThreadDataLoader(dataset, batch_size=batch_size, 
                                                  shuffle=True, num_workers=data_loader_num_worker, 
                                                  pin_memory=torch.cuda.is_available(),
                                                  use_thread_workers=True,
                                                  persistent_workers = bool(data_loader_num_worker))

    else:
        data_loader = monai.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    
    # create a validation data loader
    return_dictionary = {"data_loader" : data_loader, "initial_transform":initial_transform,
                         "input_sample":input_sample,"input_shape":input_shape,
                         "post_pred":post_pred,  "post_label" : post_label}
    return return_dictionary
