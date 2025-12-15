import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# --- CONFIGURATION ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
PARTICLE_COUNT = 2500
MAX_EXPANSION = 8.0

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- STATE MANAGEMENT ---
class State:
    def __init__(self):
        self.hand_detected = False
        self.hand_pos = np.array([0.0, 0.0, 0.0])
        self.pinch_factor = 0.5
        self.palm_openness = 0.0
        self.current_template = 0 # 0: Torus, 1: Heart, etc.

state = State()

# --- PARTICLE SYSTEM ---
class ParticleSystem:
    def __init__(self, count):
        self.count = count
        # Initialize random positions/velocities
        # Velocity is a normalized random vector
        self.velocities = np.random.uniform(-1, 1, (count, 3))
        norms = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = self.velocities / norms
        
        self.offsets = np.random.uniform(0, 100, count)
        self.positions = np.zeros((count, 3))
        self.colors = np.ones((count, 3)) # RGB
        
        self.current_scale_multiplier = 1.0

    def update(self, time, state):
        # 1. Calculate Target Global Scale (Expansion)
        # Target = 1.0 + (Openness * MaxExpansion)
        target_global_scale = 1.0 + (state.palm_openness * MAX_EXPANSION)
        
        # Smooth interpolation (Lerp)
        self.current_scale_multiplier += (target_global_scale - self.current_scale_multiplier) * 0.1
        
        # 2. Base Spread Logic
        # If hand detected, spread depends on pinch. Else default wide spread.
        if state.hand_detected:
            base_spread = 2.0 + (state.pinch_factor * 8.0)
        else:
            base_spread = 10.0
            
        target_spread = base_spread * self.current_scale_multiplier
        
        # 3. Attraction Logic
        attraction_center = state.hand_pos if state.hand_detected else np.array([0.0, 0.0, 0.0])
        attraction_strength = 0.05 if state.hand_detected else 0.01

        # 4. Color Logic (HSL to RGB conversion simplified)
        # Closed (0) = Blueish, Open (1) = Reddish
        # We'll just lerp between Blue and Red for simplicity in RGB
        # Blue: [0, 0, 1], Red: [1, 0, 0]
        hue_factor = state.pinch_factor
        color_target = np.array([hue_factor, 0.2, 1.0 - hue_factor]) # Simple gradient
        
        # Brighten based on expansion
        brightness = 0.5 + (state.palm_openness * 0.5)
        self.colors[:] = color_target * brightness

        # 5. Vectorized Position Calculation (NumPy is fast!)
        # Calculate radius for all particles
        time_offsets = (time * 0.5) + self.offsets
        sine_wave = np.sin(time_offsets * 0.7) * 0.2
        radii = target_spread * (0.8 + sine_wave)
        
        # Calculate target positions based on orbit math
        target_x = attraction_center[0] + np.sin(time_offsets) * radii * self.velocities[:, 0]
        target_y = attraction_center[1] + np.cos(time_offsets * 0.9) * radii * self.velocities[:, 1]
        target_z = attraction_center[2] + np.sin(time_offsets * 1.1) * radii * self.velocities[:, 2]
        
        targets = np.stack((target_x, target_y, target_z), axis=1)
        
        # Lerp current positions to target
        self.positions += (targets - self.positions) * attraction_strength

    def draw(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        glVertexPointer(3, GL_DOUBLE, 0, self.positions)
        glColorPointer(3, GL_DOUBLE, 0, self.colors)
        
        glDrawArrays(GL_POINTS, 0, self.count)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

# --- HELPER FUNCTIONS ---

def get_palm_openness(landmarks):
    # Landmarks: 0=Wrist, 9=MiddleFingerMCP (Palm Center)
    # Tips: 4, 8, 12, 16, 20
    palm_center = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    tips_indices = [4, 8, 12, 16, 20]
    total_dist = 0
    
    for idx in tips_indices:
        tip = np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])
        total_dist += np.linalg.norm(tip - palm_center)
        
    avg_dist = total_dist / 5.0
    
    # Thresholds (Tunable)
    closed_thresh = 0.1
    open_thresh = 0.35
    
    openness = (avg_dist - closed_thresh) / (open_thresh - closed_thresh)
    return max(0.0, min(1.0, openness))

def map_hand_to_screen(landmarks):
    # 1. Position: Palm Center (9)
    # MediaPipe is 0-1. OpenGL world is roughly -20 to 20
    x = (1.0 - landmarks[9].x) * 40 - 20 # Flip X for mirror effect
    y = (1.0 - landmarks[9].y) * 30 - 15 # Invert Y
    z = landmarks[9].z * -50
    
    # 2. Pinch: Index(8) vs Thumb(4)
    idx = np.array([landmarks[8].x, landmarks[8].y])
    thumb = np.array([landmarks[4].x, landmarks[4].y])
    dist = np.linalg.norm(idx - thumb)
    
    pinch = (dist - 0.02) / (0.15 - 0.02)
    pinch = max(0.0, min(1.0, pinch))
    
    return np.array([x, y, z]), pinch

# --- MAIN LOOP ---

def main():
    # PyGame / OpenGL Setuppy
    pygame.init()
    display = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Python Hand Gesture Particles")

    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glTranslatef(0.0, 0.0, -40) # Move camera back

    # Camera Setup
    # Use DirectShow (CAP_DSHOW) on Windows to fix initialization issues
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check if it is connected or used by another app.")
        return
    
    particle_system = ParticleSystem(PARTICLE_COUNT)
    clock = pygame.time.Clock()
    start_time = pygame.time.get_ticks()

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 2. Computer Vision (MediaPipe)
        success, image = cap.read()
        if success:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            try:
                results = hands.process(image_rgb)
            except Exception as e:
                print(f"MediaPipe Error: {e}")
                results = None

            if results and results.multi_hand_landmarks:
                state.hand_detected = True
                lm = results.multi_hand_landmarks[0].landmark
                
                # Calculate metrics
                state.hand_pos, state.pinch_factor = map_hand_to_screen(lm)
                raw_openness = get_palm_openness(lm)
                
                # Smooth openness
                state.palm_openness += (raw_openness - state.palm_openness) * 0.1
                
            else:
                state.hand_detected = False
                state.palm_openness *= 0.9 # Decay if hand lost
        else:
            print("Warning: Camera connected but returned an empty frame.")

        # 3. Update Physics
        current_time = (pygame.time.get_ticks() - start_time) / 1000.0
        particle_system.update(current_time, state)

        # 4. Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Optional: Add simple rotation to the whole scene
        # glRotatef(1, 3, 1, 1) 
        
        glPointSize(3 if state.hand_detected else 2) # Bigger points when active
        particle_system.draw()
        
        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()