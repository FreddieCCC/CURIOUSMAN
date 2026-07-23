# pac.py
import pygame
from settings import CHAR_SIZE, PLAYER_SPEED
from animation import import_sprite

class Pac(pygame.sprite.Sprite):
    def __init__(self, row, col):
        super().__init__()
        self.abs_x = (row * CHAR_SIZE)
        self.abs_y = (col * CHAR_SIZE)
        # pac animation
        self._import_character_assets()
        self.frame_index = 0
        self.animation_speed = 0.5
        base=self.animations["idle"][self.frame_index]
        self.image = pygame.transform.scale(base, (CHAR_SIZE, CHAR_SIZE))
        self.rect = self.image.get_rect(topleft = (self.abs_x, self.abs_y))
        self.mask = pygame.mask.from_surface(self.image)
        self.pac_speed = PLAYER_SPEED
        self.directions = {'left': (-PLAYER_SPEED, 0), 'right': (PLAYER_SPEED, 0), 'up': (0, -PLAYER_SPEED), 'down': (0, PLAYER_SPEED), 'idle': (0,0)}
        self.keys = {'left': pygame.K_LEFT, 'right': pygame.K_RIGHT, 'up': pygame.K_UP, 'down': pygame.K_DOWN}
        self.direction = (0, 0)
        self.radius = 2 # 5x5 grid area
        # pac status
        self.status = "idle"

    # gets all the image needed for animating specific player action
    def _import_character_assets(self):
        character_path = "assets/pac/"
        self.animations = {
            "up": [],
            "down": [],
            "left": [],
            "right": [],
            "idle": [],
        }
        for animation in self.animations.keys():
            full_path = character_path + animation
            self.animations[animation] = import_sprite(full_path)

    def _is_collide(self, x, y, walls_collide_list):
        tmp_rect = self.rect.move(x, y)
        return tmp_rect.collidelist(walls_collide_list) != -1

    
    def move_to_start_pos(self):
        self.rect.x = self.abs_x
        self.rect.y = self.abs_y

    # update with sprite/sheets
    def animate(self):
        animation = self.animations[self.status]
        # loop over frame index
        self.frame_index += self.animation_speed
        if self.frame_index >= len(animation):
            self.frame_index = 0
        topleft =self.rect.topleft
        base=animation[int(self.frame_index)]
        self.image = pygame.transform.scale(base, (CHAR_SIZE, CHAR_SIZE))
        self.rect = self.image.get_rect(topleft = topleft)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.animate()

    # the agent will have its location in the gridspace
    def get_pos(self):
        return (self.rect.x // CHAR_SIZE, self.rect.y // CHAR_SIZE)

    def step(self, action, walls_collide_list):
        #safe return
        attempted=True
        collided=False
        target_tile=None

        cx,cy=self.get_pos()
        #IDLE principle
        if action == "idle":
            self.status="idle"
            return True, False, (cx,cy)
        #Invalid principle
        if action not in self.directions:
            return False,False,None
        dx,dy=self.directions[action]
        gx=1 if dx>0 else(-1 if dx<0 else 0)
        gy=1 if dy>0 else(-1 if dy<0 else 0)
        target_tile=(cx+gx,cy+gy)

        #collision check
        collided = self._is_collide(dx, dy, walls_collide_list)
        if not collided:
            self.rect.move_ip((dx, dy))
            self.status=action
            self.direction=(dx,dy)
        return attempted, collided, target_tile
    