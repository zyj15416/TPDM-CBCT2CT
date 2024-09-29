from Src import model_v1, model_v2, model_v3, model_v4, model_v5, model_v6, model_v8, models


class config:
    model_config = {
        0: {'model': model_v1.model_S(),
            'loss': False},
        1: {'model': model_v2.model_S(),
            'loss': False},
        2: {'model': model_v3.model_S(),
            'loss': False},
        3: {'model': model_v4.model_S(),
            'loss': True},
        4: {'model': model_v5.model_S(),
            'loss': False},
        5: {'model': model_v6.model_S(),
            'loss': False},
        6: {'model': model_v1.model_S(),
            'loss': True},
        7: {'model': model_v8.model_S(),
            'loss': False},
        8: {'model': model_v2.model_S(),
            'loss': True},
        9: {'model': model_v3.model_S(),
            'loss': True},
        10: {'model': models.model_S(),
             'loss': False},
        11: {'model': model_v5.model_S(),
             'loss': True},
        12: {'model': model_v6.model_S(),
             'loss': True},
        13: {'model': model_v8.model_S(),
             'loss': True}
    }
