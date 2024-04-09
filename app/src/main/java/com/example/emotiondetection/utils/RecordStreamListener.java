package com.example.emotiondetection.utils;

public interface RecordStreamListener {
    void recordOfByte(byte[] data, int begin, int end);
}
