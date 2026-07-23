# display.py
import pygame
from settings import WIDTH, HEIGHT, CHAR_SIZE
pygame.font.init()

class Display:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("ubuntumono", CHAR_SIZE)
        self.game_over_font = pygame.font.SysFont("dejavusansmono", 48)
        self.text_color = pygame.Color("crimson")
  
    def show_episode(self, episode):
        episode = self.font.render(f'Episode: {episode}', True, self.text_color)
        self.screen.blit(episode, (CHAR_SIZE // 2, HEIGHT + (CHAR_SIZE // 2)))

    # add game over message
    def game_over(self):
        message = self.game_over_font.render(f'GAME OVER!!', True, pygame.Color("chartreuse"))
        instruction = self.font.render(f'Press "R" to Restart', True, pygame.Color("aqua"))
        self.screen.blit(message, ((WIDTH // 4), (HEIGHT // 3)))
        self.screen.blit(instruction, ((WIDTH // 4), (HEIGHT // 2)))