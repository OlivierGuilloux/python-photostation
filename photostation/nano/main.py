import sys
sys.path.insert(0, '../')
import time, os
from photostation import PhotoStationService, SynologyException
import requests
import detect
import config


root_folder = sys.argv[1]
# login to Photo Station and set up root album
service = PhotoStationService(config.photostation_uri, root_folder)
tags_list = service.list_tags().get('tags', [])

def get_tag_from_name(tag_name):
    for tag in tags_list :
        if tag['name'] == tag_name:
            return tag['id']
    return None

def get_next_unknown_tag(used_tags_id, i=0):
    #print(f'get_next_unknown_tag {i}')
    tag_id = get_tag_from_name(f'_inc_{i}')
    if tag_id is not None and tag_id not in used_tags_id:
        return tag_id
    elif tag_id is not None and tag_id in used_tags_id:
        return get_next_unknown_tag(used_tags_id, i+1)
    else:
        tag_id = service.create_tag(f'_inc_{i}')
        # XXX Prevent load on photostation
        time.sleep(5)
        return tag_id['id']

for item, item_type in service.root_album.items.items():
    if item_type.filetype != 'photo':
        continue
    if item_type.tags['tags'] != []:
        continue
    #print(item_type)
    #print(item_type.tags)
    filename = item_type.download()
    # appliquer la reconnaissance
    print(f'Apply face detection on {filename}')
    boxes = detect.apply_on_image(filename)
    # obtenir la boite
    # tag avec confirme à faut
    used_tags_id = []
    for box in boxes:
        #print(box)
        if box['cat'] != '':
            # Obtenir le tag_id
            tag_id = get_tag_from_name(box['cat'])
            # Prevent PHOTOSTATION_PHOTO_TAG_DUPLICATE
            if tag_id in used_tags_id or tag_id is None:
                tag_id = get_next_unknown_tag(used_tags_id)
                used_tags_id.append(tag_id)
        else:
            tag_id = get_next_unknown_tag(used_tags_id)
        item_type.ps_tag.add_people_tag(tag_id, box['x'], box['y'], box['w'], box['h'])
        item_type.ps_tag.people_tag_confirm(tag_id, 'false')
        if tag_id not in used_tags_id:
            used_tags_id.append(tag_id)
        #print(f' done {len(used_tags_id)} tag(s) added')
    os.remove(filename)
