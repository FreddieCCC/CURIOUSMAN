from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class PirateLog:
    path:str="run_log.csv"
    flush_every:int=1

    def __post_init__(self):
        self._f = None
        self._w = None
        self._rows_since_flush:int=0

        header = (not os.path.exists(self.path)) or (os.path.getsize(self.path)==0)

        self._f=open(self.path, "a", newline="", encoding="utf-8")
        self._w=csv.DictWriter(self._f, fieldnames=self.fieldnames(), extrasaction="raise")
        if header:
            self._w.writeheader()
            self._f.flush()

    @staticmethod
    def fieldnames():
        return [
            "wall_time",
            "episode",
            "step_in_episode",
            "x",
            "y",
            "action",
            "attempted",
            "target_x",
            "target_y",
            "best_score",
            "chosen_score",
            "mean_prediction_error",
            "local_uncertainty",
            "frontier_ignorance",
            "ghost_certainty",
            "tiles_explored",
            "new_tiles",
        ]
    
    def log_step(self, *, episode,step_in_episode, x,y, action, attempted, target_tile, best_score, chosen_score,mean_prediction_error, local_uncertainty, frontier_ignorance, ghost_certainty, tiles_explored, new_tiles):
        tx,ty = target_tile if target_tile is not None else (None, None)
        row = {
            "wall_time": time.time(),
            "episode": episode,
            "step_in_episode": step_in_episode,
            "x": x,
            "y": y,
            "action": action,
            "attempted": int(bool(attempted)),
            "target_x": tx,
            "target_y": ty,
            "best_score": best_score,
            "chosen_score": chosen_score,
            "mean_prediction_error": mean_prediction_error,
            "local_uncertainty": local_uncertainty,
            "frontier_ignorance": frontier_ignorance if frontier_ignorance is not None else None,
            "ghost_certainty": ghost_certainty if ghost_certainty is not None else None,
            "tiles_explored": tiles_explored,
            "new_tiles": new_tiles
        }
        self._w.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 0 and self._rows_since_flush >= self.flush_every:
            self._f.flush()
            self._rows_since_flush = 0
        
    def close(self):
        try:
            if self._f:
                self._f.flush()
                self._f.close()
        finally:
            self._f = None
            self._w = None 