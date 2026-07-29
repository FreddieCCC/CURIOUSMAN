#CONDITIONS:
    #DECAY_ON: Epistemic weight decay is on
    #Decay_pff: Epistemic weight decay is off
    #C3_RANDOM: random action selection
    #C4_PURE_PE: pure PE seeking, no decay

import os
import sys
import random
import argparse
from datetime import datetime
import runpy
import world

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()

from settings import WIDTH, HEIGHT, NAV_HEIGHT, CHAR_SIZE, MAP, PLAYER_SPEED
from world import World
from pirate_log import PirateLog

class Conditions(World):
    def __init__(self, screen, condition="decay_on",seed=None):
        self.condition = condition
        self._seed = seed if seed is not None else 0
        if seed is not None:
            random.seed(seed)
        super().__init__(screen)

        self.log.close()
        ts= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join("logs",f"run_log_{condition}_seed{self._seed}_{ts}.csv")
        self.log = PirateLog(path=log_path, flush_every=30)

    def choose_action(self):
        if self.condition == "c1_random":
            actions = ["up", "down", "left", "right", "idle"]
            chosen = random.choice(actions)
            return chosen, 0.0, 0.0
        return super().choose_action()

    def action_score(self, action):
        if self.condition == "c5_pure_pe":
            pac = self.player.sprite
            if action not in pac.directions:
                return -1e9
            dx, dy = pac.directions[action]
            if pac._is_collide(dx, dy, self.walls_collide_list):
                return -1e9
            next_rect = pac.rect.move(dx, dy)
            nx = next_rect.x // CHAR_SIZE
            ny = next_rect.y // CHAR_SIZE
            return self.expected_uncertainty(nx, ny, pac.radius*2)

        if self.condition == "decay_on":
            pac = self.player.sprite
            if action not in pac.directions:
                return -1e9
            dx, dy = pac.directions[action]
            if pac._is_collide(dx, dy, self.walls_collide_list):
                return -1e9
            next_rect = pac.rect.move(dx, dy)
            nx = next_rect.x // CHAR_SIZE
            ny = next_rect.y // CHAR_SIZE
            r = pac.radius
            epistemic = self.expected_uncertainty(nx, ny, r * 2)
            sf = self.seen_fraction()
            seen_decay = self.epistemic_decay * sf
            weight = max(self.epistemic_weight_min, self.epistemic_weight * (1 - seen_decay))  # decay epistemic weight as more of the map is seen
            #ghost certainty
            ghost_term = 0
            ignorance, visible, ghost_pos = self.ghost_ignorance()
            if visible and ghost_pos is not None and self.ghost_certainty < self.ghost_follow_thresh:
                gx,gy =ghost_pos
                d=abs(gx - nx) + abs(gy - ny)
                proximity = 1/(d+1)
                ghost_term = self.ghost_follow_weight * proximity
                return (weight * epistemic) + ghost_term
            return weight * epistemic

        if self.condition == "decay_off":
            pac = self.player.sprite
            if action not in pac.directions:
                return -1e9
            dx, dy = pac.directions[action]
            if pac._is_collide(dx, dy, self.walls_collide_list):
                return -1e9
            next_rect = pac.rect.move(dx, dy)
            nx = next_rect.x // CHAR_SIZE
            ny = next_rect.y // CHAR_SIZE
            r = pac.radius
            epistemic = self.expected_uncertainty(nx, ny, r * 2)
            ghost_term = 0
            ignorance, visible, ghost_pos = self.ghost_ignorance()
            if visible and ghost_pos is not None and self.ghost_certainty < self.ghost_follow_thresh:
                gx,gy =ghost_pos
                d=abs(gx - nx) + abs(gy - ny)
                proximity = 1/(d+1)
                ghost_term = self.ghost_follow_weight * proximity
                return epistemic + ghost_term
            return epistemic
        
        return super().action_score(action)


    def update(self):
        if not self.game_over:
            action, best_score, chosen_score = self.choose_action()
            attempted, collided, target_tile = self.player.sprite.step(
                action, self.walls_collide_list
            )
            new_tiles, mean_pe = self.update_belief()
            x, y = self.player.sprite.get_pos()
            lu = self.local_uncertainty()
            fi = self.frontier_ignorance()
            sf = self.seen_fraction()
            self.log.log_step(
                episode=self.episode,
                step_in_episode=self.steps_in_episode,
                x=x, y=y,
                action=action,
                attempted=attempted,
                target_tile=target_tile,
                best_score=best_score,
                chosen_score=chosen_score,
                mean_prediction_error=mean_pe,
                local_uncertainty=lu,
                frontier_ignorance=fi,
                ghost_certainty=self.ghost_certainty,
                tiles_explored=sf,
                new_tiles=new_tiles,
            )
            self.t += 1
            if attempted and target_tile is not None:
                tx, ty = target_tile
                if 0 <= tx < self.W and 0 <= ty < self.H:
                    o = 1 if collided else 0
                    p = float(self.p_wall[tx][ty])
                    updated = p + self.wall_rate * (o - p)
                    self.p_wall[tx][ty] = min(1, max(0, updated))
            self.steps_in_episode += 1
            if self.steps_in_episode >= self.max_steps_per_episode:
                self.episode_lengths.append(self.steps_in_episode)
                self.episode += 1
                self.steps_in_episode = 0
                self.reset_episode()
            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0
            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                    self.episode_lengths.append(self.steps_in_episode)
                    self.ghost_certainty = (
                        self.ghost_certainty + self.ghost_rate * (1 - self.ghost_certainty)
                    )
                    self.episode += 1
                    if self.episode >= self.max_episodes:
                        self.game_over = True
                        break
                    self.steps_in_episode = 0
                    self.reset_episode()
                    break
        self._check_game_state()

def run_condition(condition,seed,screen):
    print(f" [seed={seed}] and condition: {condition}")
    world = Conditions(screen, condition=condition, seed=seed)
    print(f"Condition: {world.condition}, choose_action is: {world.choose_action}")
    step = 0
    while not world.game_over:
        world.update()
        pygame.event.pump()
        step += 1
        if step % 50000 == 0:
            print(f" [seed={seed}] and condition: {condition} step: {step}")
    world.log.close()
    print(f" [seed={seed}] and condition: {condition} finished after {step} steps, {world.episode} episodes, {sum(world.episode_lengths)} total steps in episodes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["decay_on", "decay_off", "c1_random", "c5_pure_pe"], help="List of conditions to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=3, help="List of seeds to run")
    args = parser.parse_args()

    screen = pygame.display.set_mode((WIDTH, HEIGHT + NAV_HEIGHT))

    for condition in args.conditions:
        print(f"\n=== Condition: {condition} ===")
        for s in range(args.seeds):
            seed = 1 + s
            run_condition(condition, seed, screen)

    print("\nAll conditions and seeds completed. Logs saved in the 'logs' directory.")
    pygame.quit()

if __name__ == "__main__":
    main()
    print("\nCreating graphs ...")
    runpy.run_path("plot.py")

