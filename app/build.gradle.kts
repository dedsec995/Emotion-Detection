plugins {
    alias(libs.plugins.androidApplication)
}



android {
    namespace = "com.example.emotiondetection"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.emotiondetection"
        minSdk = 23
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.navigation.fragment)
    implementation(libs.navigation.ui)
    implementation("net.sf.supercsv:super-csv:2.4.0")
    implementation("androidx.camera:camera-camera2:1.0.0-alpha05")
    implementation("androidx.camera:camera-core:1.0.0-alpha05")
    implementation("org.pytorch:pytorch_android_lite:1.13.1")
    implementation("org.pytorch:pytorch_android_torchvision_lite:1.13.1")
    implementation("com.google.android.gms:play-services-vision:20.1.3")


    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}