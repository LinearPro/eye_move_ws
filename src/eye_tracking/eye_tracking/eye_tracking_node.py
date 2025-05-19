#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import os
import cv2
import pygame
import numpy as np
from pygame.locals import *
from cv_bridge import CvBridge
import face_recognition
import time
import threading
from queue import Queue
import re

class EyeTrackingNode(Node):
    def __init__(self):
        super().__init__('eye_tracking_node')
        
        # 初始化参数
        self.image_dir = "/home/lingp/Documents/get_image_recognize/image_raw"
        self.output_dir = "/home/lingp/Documents/get_image_recognize/output"
        self.max_process_time = 0.1  # 目标处理时间(秒)
        self.remove_processed = False  # 是否删除已处理的图片文件
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化pygame
        pygame.init()
        self.screen_width = 1500
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("眼睛随动系统(带运行时间统计)")
        
        # 加载眼睛图片
        try:
            self.eye1 = pygame.image.load("/home/lingp/Documents/eyee/eye1.png").convert_alpha()
            self.eye2 = pygame.image.load("/home/lingp/Documents/eyee/eye2.png").convert_alpha()
            self.eye3 = pygame.image.load("/home/lingp/Documents/eyee/eye3.png").convert_alpha()
            
            # 统一缩放眼睛图片
            self.eye1 = pygame.transform.scale(self.eye1, (600, 600))
            self.eye2 = pygame.transform.scale(self.eye2, (600, 600))
            self.eye3 = pygame.transform.scale(self.eye3, (600, 600))
        except Exception as e:
            self.get_logger().error(f"加载眼睛图片失败: {e}")
            raise
        
        # 获取图片矩形对象
        self.eye1_rect = self.eye1.get_rect()
        self.eye2_rect = self.eye2.get_rect()
        self.eye3_rect = self.eye3.get_rect()
        
        # 设置眼睛初始位置(右侧区域)
        self.eye1_rect.center = (self.screen_width//4*3, self.screen_height//2)
        self.eye2_rect.center = (self.screen_width//4*3, self.screen_height//2)
        self.eye3_rect.center = (self.screen_width//4*3, self.screen_height//2)
        
        # 眼睛移动参数
        self.max_eye_movement = 50  # 减小移动范围
        self.face_offset = (0, 0)
        self.current_face_img = None
        
        # OpenCV桥接
        self.bridge = CvBridge()
        
        # 线程安全队列
        self.image_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        
        # 图片列表和已处理记录
        self.image_files = []
        self.processed_files = set()  # 已处理的文件集合
        self.image_files_lock = threading.Lock()
        
        # 启动人脸检测线程
        self.detection_thread = threading.Thread(target=self._face_detection_worker, daemon=True)
        self.detection_thread.start()
        
        # 创建定时器处理图像
        self.timer = self.create_timer(0.05, self.process_next_image)  # 20fps
        
        # 创建定时器刷新图片目录 (每0.1秒)
        self.refresh_timer = self.create_timer(0.1, self.refresh_image_list)
        
        # 初始加载图片列表
        self.refresh_image_list()
        self.get_logger().info("眼睛追踪节点已启动")
    
    def refresh_image_list(self):
        """刷新图片目录中的文件列表，排除已处理的文件"""
        try:
            # 获取所有符合条件的文件
            files = [f for f in os.listdir(self.image_dir) 
                    if f.startswith('image_') and f.endswith('.jpg')]
            
            # 定义排序键函数：提取文件名中的数字部分
            def get_number(filename):
                match = re.search(r'image_(\d+)\.jpg', filename)
                return int(match.group(1)) if match else 0
            
            # 按数字顺序排序并排除已处理的
            new_files = sorted(
                [f for f in files if f not in self.processed_files],
                key=get_number
            )
            
            # 线程安全地更新图片列表
            with self.image_files_lock:
                # 只添加不在当前列表中的新文件
                current_set = set(self.image_files)
                added_files = [f for f in new_files if f not in current_set]
                
                if added_files:
                    self.image_files.extend(added_files)
                    self.get_logger().info(f"发现 {len(added_files)} 张新图片，待处理总数: {len(self.image_files)}")
        except Exception as e:
            self.get_logger().error(f"刷新图片列表时出错: {e}")
    
    def _face_detection_worker(self):
        """人脸检测线程工作函数"""
        while True:
            cv_image = self.image_queue.get()
            if cv_image is None:  # 终止信号
                break
            
            try:
                # 保持原始图像用于显示
                display_img = cv_image.copy()
                
                # 缩小图像加速处理
                small_img = cv2.resize(cv_image, (0,0), fx=0.5, fy=0.5)
                rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                
                # 检测人脸
                face_locations = face_recognition.face_locations(rgb_small)
                
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    # 在原始图像上绘制绿色框
                    cv2.rectangle(display_img, 
                                (left*2, top*2), 
                                (right*2, bottom*2), 
                                (0, 255, 0), 2)
                    
                    # 计算归一化偏移量
                    img_height, img_width = small_img.shape[:2]
                    offset_x = ((left + right)/2 - img_width/2) / (img_width/2)
                    offset_y = ((top + bottom)/2 - img_height/2) / (img_height/2)
                    
                    self.result_queue.put((offset_x, offset_y, display_img))
                else:
                    self.result_queue.put((0, 0, display_img))
                    
            except Exception as e:
                self.get_logger().error(f"人脸检测出错: {e}")
                self.result_queue.put((0, 0, cv_image))
    
    def process_next_image(self):
        """处理下一帧图像"""
        with self.image_files_lock:
            if not self.image_files:
                return
            
            # 获取当前要处理的图片
            current_image = self.image_files[0]
            image_path = os.path.join(self.image_dir, current_image)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                raise ValueError("无法读取图像")
                
            # 提交到人脸检测线程
            if self.image_queue.empty():
                self.image_queue.put(cv_image.copy())
            
            # 尝试获取最新结果
            try:
                offset_x, offset_y, face_img = self.result_queue.get_nowait()
                self.face_offset = (offset_x, offset_y)
                self.current_face_img = face_img
            except:
                pass  # 保持上次结果
            
            # 更新眼睛位置
            self._update_eye_positions()
            
            # 渲染所有内容
            self._render_all()
            
            # 保存结果
            if self.current_face_img is not None:
                output_path = os.path.join(self.output_dir, f"result_{current_image}")
                cv2.imwrite(output_path, self.current_face_img)
            
            # 计算并显示处理时间
            process_time = (time.time() - start_time) * 1000  # 转换为毫秒
            self.get_logger().info(f"处理完成: {current_image} [耗时: {process_time:.2f}ms]")
            
            # 检查处理时间是否超时
            if process_time > self.max_process_time * 1000:
                self.get_logger().warning(f"处理超时: {process_time:.2f}ms (目标: {self.max_process_time*1000:.2f}ms)")
            
            # 标记为已处理
            with self.image_files_lock:
                if self.image_files and self.image_files[0] == current_image:
                    self.processed_files.add(current_image)
                    self.image_files.pop(0)
                    
                    # 可选：删除已处理的图片文件
                    if self.remove_processed:
                        try:
                            os.remove(image_path)
                            self.get_logger().info(f"已删除原文件: {current_image}")
                        except Exception as e:
                            self.get_logger().error(f"删除文件失败: {current_image} - {e}")
            
        except Exception as e:
            self.get_logger().error(f"处理图像 {current_image} 时出错: {e}")
            with self.image_files_lock:
                if self.image_files and self.image_files[0] == current_image:
                    self.image_files.pop(0)
    
    def _update_eye_positions(self):
        """更新眼睛位置"""
        offset_x, offset_y = self.face_offset
        
        # 计算移动量
        move_x = -offset_x * self.max_eye_movement
        move_y = offset_y * self.max_eye_movement
        
        # 更新眼球和虹膜位置
        self.eye1_rect.center = (
            self.screen_width//4*3 + move_x,
            self.screen_height//2 + move_y
        )
        self.eye2_rect.center = (
            self.screen_width//4*3 + move_x * 0.7,
            self.screen_height//2 + move_y * 0.7
        )
        
        # 边界检查
        for rect in [self.eye1_rect, self.eye2_rect]:
            if rect.left < self.screen_width//2:
                rect.left = self.screen_width//2
            if rect.right > self.screen_width:
                rect.right = self.screen_width
            if rect.top < 0:
                rect.top = 0
            if rect.bottom > self.screen_height:
                rect.bottom = self.screen_height
    
    def _render_all(self):
        """渲染所有元素"""
        # 清屏
        self.screen.fill((0, 0, 0))
        
        # 显示原始图像和人脸框
        if self.current_face_img is not None:
            # 转换OpenCV图像为Pygame表面
            face_img_rgb = cv2.cvtColor(self.current_face_img, cv2.COLOR_BGR2RGB)
            face_img_surface = pygame.surfarray.make_surface(
                np.rot90(face_img_rgb)  # 需要旋转90度
            )
            # 缩放图像以适应左侧区域
            face_img_surface = pygame.transform.scale(
                face_img_surface, 
                (self.screen_width//2, self.screen_height)
            )
            self.screen.blit(face_img_surface, (0, 0))
        
        # 绘制眼睛图片(从底层到顶层)
        self.screen.blit(self.eye3, self.eye3_rect)  # 眼白
        self.screen.blit(self.eye2, self.eye2_rect)  # 虹膜
        self.screen.blit(self.eye1, self.eye1_rect)  # 眼球
        
        # 更新显示
        pygame.display.flip()
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = EyeTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
