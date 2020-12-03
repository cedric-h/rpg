import json
import bpy

with open('./map.json', 'w') as fp:
    o = {
        'trees': [
            o.location[:2]
                for o in bpy.data.objects
                if o.name.startswith('Tree')
        ],
        'brambles': [
            o.location[:2]
                for o in bpy.data.objects
                if o.name.startswith('Bramble')
        ],
        'circles': [
            { 'pos': o.location[:2], 'radius': o.scale[0] / 2.0 }
                for o in bpy.data.objects
                if o.name.startswith('Circle')
        ],
    };
    
    splines = [
        "rocky_enter", "rocky_hide", "rocky_leave",
            "rocky_bounce1", "rocky_bounce2",
        "melee_enter", "ranged_enter",
    ]
    
    for spline in splines:
        bps = [
            o.data.splines[0].bezier_points
                for o in bpy.data.objects
                if o.name == spline
        ][0]
        o[spline] = [
            {
                'left': bp.handle_left[:2],
                'pos': bp.co[:2],
                'right': bp.handle_right[:2]
            }
                for bp in bps
        ]
    
    json.dump(o, fp)