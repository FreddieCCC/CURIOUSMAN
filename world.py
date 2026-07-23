import random
import pygame
import time
import math
import os
from datetime import datetime
from settings import HEIGHT, WIDTH, NAV_HEIGHT, CHAR_SIZE, MAP, PLAYER_SPEED
from pac import Pac
from cell import Cell
from ghost import Ghost
from display import Display
from pirate_log import PirateLog

class World:
    def __init__(self, screen):
        self.screen = screen
        self.player = pygame.sprite.GroupSingle()
        self.ghosts = pygame.sprite.Group()
        self.ghost_certainty = 0                                              #agent's belief about ghost's "purpose" 
        self.ghost_rate = 0.2                                                 # learning rate per terminal encounter
        self.ghost_follow_thresh = 0.8                                        #max certainty to stop follow ghost
        self.ghost_follow_weight = 5                                          #action selection between ghost and wall exploration
        self.walls = pygame.sprite.Group()
        self.wall_rate = 0.2                                                  #learning rate for wall belief update
        self.display = Display(self.screen)
        self.game_over = False
        self.reset_pos = False
        self.episode = 1
        self.steps_in_episode = 0
        self.max_episodes = 21                                               #likely for fulfilling experimental goals
        self.max_steps_per_episode = 10000
        self.episode_lenghts =[]
        self.H = len(MAP)
        self.W = len(MAP[0])
        self.p_wall = [[0.5 for _ in range(self.H)] for _ in range(self.W)] #agent's belief that there is a wall in that tile//0.5 as prior
        self.seen = [[0 for _ in range(self.H)] for _ in range(self.W)]     # tiles seen by the agent
        self.epistemic_weight = 1                                             #weight for expected prediction-error
        self.epistemic_decay = 0.95                                            # per-episode decay
        self.epistemic_weight_min = 0.05                                         #so it never becomes zero
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join("logs", f"run_log_{ts}.csv")
        self.log = PirateLog(log_path, flush_every=30)
        self.t = 0 #step counter for logging
        self._generate_world()
    
        # create and add player to the screen
    def _generate_world(self):
        # renders obstacle from the MAP table
        for y_index, col in enumerate(MAP):
            for x_index, char in enumerate(col):
                if char == "1":	# for walls
                    self.walls.add(Cell(x_index, y_index, CHAR_SIZE, CHAR_SIZE))
                # for Ghosts's starting position
                elif char == "s":
                    self.ghosts.add(Ghost(x_index, y_index, "skyblue"))
                elif char == "P":	# for PacMan's starting position 
                    self.player.add(Pac(x_index, y_index))
                else:
                    pass        
        self.walls_collide_list = [wall.rect for wall in self.walls.sprites()]

    def reset_episode(self):
        self.player.sprite.move_to_start_pos()
        for ghost in self.ghosts:
            ghost.move_to_start_pos()

        # displays nav
    def _dashboard(self):
        nav = pygame.Rect(0, HEIGHT, WIDTH, NAV_HEIGHT)
        pygame.draw.rect(self.screen, pygame.Color("cornsilk4"), nav)

        self.display.show_episode(self.episode)

    def _check_game_state(self):
        # checks if game over
        if self.episode >= self.max_episodes:
            self.game_over = True

    def update(self):
        if not self.game_over:
            # player movement
            new_tiles, mean_pe =self.update_belief() # update agent's beliefs based on observation; mean predictive error
            action, best_score, chosen_score = self.choose_action() # choose action based on current beliefs
            attempted, collided,target_tile = self.player.sprite.step(action,self.walls_collide_list) # execute action (by itself)
            x,y = self.player.sprite.get_pos()
            lu = self.local_uncertainty()
            fi = self.frontier_ignorance()
            sf = self.seen_fraction()
            #log
            self.log.log_step(
                episode=self.episode,
                step_in_episode=self.steps_in_episode,
                x=x,
                y=y,
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
                new_tiles=new_tiles
            )
            self.t += 1
             # update belief based on action outcome
            if attempted and target_tile is not None:
                tx,ty=target_tile
                if 0<=tx<self.W and 0<=ty<self.H:
                    o = 1 if collided else 0
                    p=float(self.p_wall[x][y])
                    updated=p+ self.wall_rate*(o-p)
                    self.p_wall[tx][ty]=min(1,max(0,updated))
            self.steps_in_episode += 1
            if self.steps_in_episode >= self.max_steps_per_episode:
                self.episode_lenghts.append(self.steps_in_episode)
                self.episode += 1
                self.epistemic_weight = max(self.epistemic_weight_min, self.epistemic_weight*self.epistemic_decay)
                self.steps_in_episode = 0
                self.reset_episode()
            # teleporting to the other side of the map
            if self.player.sprite.rect.right <= 0:
                self.player.sprite.rect.x = WIDTH
            elif self.player.sprite.rect.left >= WIDTH:
                self.player.sprite.rect.x = 0
            # PacMan bumping into ghosts
            for ghost in self.ghosts.sprites():
                if self.player.sprite.rect.colliderect(ghost.rect):
                        self.episode_lenghts.append(self.steps_in_episode)
                        self.ghost_certainty = self.ghost_certainty + self.ghost_rate * (1 - self.ghost_certainty)
                        self.episode += 1
                        self.epistemic_weight=max(self.epistemic_weight_min, self.epistemic_weight*self.epistemic_decay)
                        if self.episode >= self.max_episodes:
                            self.game_over = True
                            break
                        self.steps_in_episode = 0
                        self.reset_episode()
                        break
        
        self._check_game_state()
        # rendering
        [wall.update(self.screen) for wall in self.walls.sprites()]
        [ghost.update(self.walls_collide_list) for ghost in self.ghosts.sprites()]
        self.ghosts.draw(self.screen)
        self.player.update()
        self.player.draw(self.screen)
        self.display.game_over() if self.game_over else None
        self._dashboard()
        # reset Pac and Ghosts position after PacMan get captured
        if self.reset_pos and not self.game_over:
            [ghost.move_to_start_pos() for ghost in self.ghosts.sprites()]
            self.player.sprite.move_to_start_pos()
            self.player.sprite.status = "idle"
            self.player.sprite.direction = (0,0)
            self.reset_pos = False
        # for restart button
        if self.game_over:
            pressed_key = pygame.key.get_pressed()
            if pressed_key[pygame.K_r]:
                self.game_over = False
                self.restart_level()
        
    #this produces the perception map for the agent
    def get_location(self):
        pac = self.player.sprite
        px, py = pac.get_pos()
        r = pac.radius

        obs = []
        for y in range(max(0, py-r),min(self.H,py+r+1)):
            for x in range(max(0, px-r),min(self.W,px+r+1)):
                obs.append((x,y,MAP[y][x]))
        return obs
    
    #entropy of a tile (uncertainty per tile)
    def tile_entropy(self,p):
        eps = 1e-9
        p = min(max(p,eps),1-eps)
        return -p*math.log(p)-(1-p)*math.log(1-p)
    
    #local uncertainty (entropy) around the agent (instead of total map coverage)
    def local_uncertainty(self):
        pac = self.player.sprite
        px, py = pac.get_pos()
        r = pac.radius

        total_entropy = 0
        for y in range(max(0,py-r),min(self.H,py+r+1)):
            for x in range(max(0,px-r),min(self.W,px+r+1)):
                total_entropy += self.tile_entropy(self.p_wall[x][y])
        return total_entropy
    
    # update beliefs based on observation
    def update_belief(self):
        obs = self.get_location()
        new_tiles =0
        mean_pe=0
        pe_sum = 0 #predictive error
        pe_n = 0

        for x,y, char in obs:
            if self.seen[x][y]==0:
                new_tiles+=1
            
            #observation: wall=1, free=0
            o = 1 if char=='1' else 0
            #predictive belief BEFORE update
            p = float(self.p_wall[x][y])
            #prediction error 
            pe=abs(o-p)
            pe_sum+=pe
            pe_n+=1
            #predictive update (rescorla wagner model)
            #p = p + learning rate * (o-p)
            updated=p+self.wall_rate*(o-p)
            self.p_wall[x][y]=min(1,max(0,updated))
            #mark as seen
            self.seen[x][y] =1

        mean_pe = (pe_sum/pe_n) if pe_n >0 else 0
        return new_tiles, mean_pe


    def expected_uncertainty(self, nx,ny,r):
        total = 0
        count = 0
        for y in range(max(0,ny-r), min(self.H,ny+r+1)):
            for x in range(max(0,nx-r), min(self.W, nx+r+1)):
                p=float(self.p_wall[x][y])
                total += p*(1-p)
                count += 1
        return (total/count) if count >0 else 0
    #UNKNOWN TILE == seen[x][y] == 0
    
    def action_score(self,action):
        pac = self.player.sprite
        if action not in pac.directions:
            return -1e9
        dx,dy = pac.directions[action]
        if pac._is_collide(dx,dy,self.walls_collide_list):
            return -1e9 #if collision, low score
        
        next_rect = pac.rect.move(dx,dy)
        nx =next_rect.x // CHAR_SIZE
        ny =next_rect.y // CHAR_SIZE
        r =pac.radius

        #epistemic value from predictive beliefs: expected uncertainty in the next observation window
        epistemic = self.expected_uncertainty(nx,ny,r)

        #adjust gain based on ghost certainty

        ignorance, visible, ghost_pos = self.ghost_ignorance()
        if visible and ghost_pos is not None and self.ghost_certainty < self.ghost_follow_thresh:
            gx,gy =ghost_pos
            d=abs(gx - nx) + abs(gy - ny)
            proximity = 1/(d+1)
            return self.ghost_follow_weight*proximity 
        return self.epistemic_weight*epistemic

    def choose_action(self):
        actions = ["left","right","up","down","idle"]
        scored = []
        for action in actions:
            score=self.action_score(action)
            scored.append((score,action))

        best_score = max(score for score,_ in scored)
        best_actions = [action for score,action in scored if score == best_score]
        chosen_action = random.choice(best_actions)
        return chosen_action, best_score, best_score
    
    def frontier_ignorance(self):
        pac=self.player.sprite
        px,py=pac.get_pos()
        r=pac.radius
        count=0
        for y in range(max(0,py-2*r),min(self.H,py+2*r+1)):
            for x in range(max(0,px-2*r),min(self.W,px+2*r+1)):
                if self.seen[x][y]==0:
                    count+=1
        return count
    
    def ghost_ignorance(self):
        if len(self.ghosts.sprites())==0:
            return 0, False,None
        pac=self.player.sprite
        px,py=pac.get_pos()
        r=pac.radius
        
        ghost = self.ghosts.sprites()[0]
        gx,gy = ghost.get_pos()

        visible = (abs(gx-px)<=r) and (abs(gy-py)<=r)
        ignorance = 1 - self.ghost_certainty

        return ignorance, visible, (gx,gy)
    
    def seen_fraction(self):
        total=self.H*self.W
        seen=sum(sum(row) for row in self.seen)
        return seen/total