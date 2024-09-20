def get_car(license_plate, vehicle_track_ids):
    """
    return a tuple containing the vehicle coordinates (x1,, y1, x2, y2)
    and ID retrieved from the vehicle based on the license plate coordinates
    """
    return 0,0,0,0,0

def read_license_plate(license_plate_crop):
    
    """
    read the license plate text from the given cropped image.
    return tuple containing the formatted license plate text and its confidence score.
    """
    
    return None, None

def write_result_csv_file(results, output_path):
    """
    write the results to a csv file.
    """
    
    with open(output_path, 'w') as f:
        
        # the header of the file 
        f.write('{},{},{},{},{},{},{}\n'.format('frame_number',
                                                'car_id',
                                                'car_bbox',
                                                'license_plate_bbox',
                                                'license_plate_bbox_score',
                                                'license_number',
                                                'license_number_score'))

        # iterate each frame of the video
        for frame_number in results.keys():
            # iterate each vehicle in each frame
            for car_id in results[frame_number].keys():
                print(results[frame_number][car_id]) # for debugging 
                
                # check if this information exist 
                if 'car' in results[frame_number][car_id].keys() and \
                    'license_plate' in results[frame_number][car_id].keys() and \
                    'text' in results[frame_number][car_id]['license_plate'].keys():
                        f.write('{},{},{},{},{},{},{}\n'.format(frame_number,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][car_id]['car']['bbox'][0],
                                                                results[frame_number][car_id]['car']['bbox'][1],
                                                                results[frame_number][car_id]['car']['bbox'][2],
                                                                results[frame_number][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_number][car_id]['license_plate']['bbox'][0],
                                                                results[frame_number][car_id]['license_plate']['bbox'][1],
                                                                results[frame_number][car_id]['license_plate']['bbox'][2],
                                                                results[frame_number][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_number][car_id]['license_plate']['bbox_score'],
                                                            results[frame_number][car_id]['license_plate']['text'],
                                                            results[frame_number][car_id]['license_plate']['text_score'])
                            )
        f.close()